#!/usr/bin/env python3
"""
Анализ фотографии агарозного геля: сегментация полос моделью GelGenie (U-Net, ONNX),
разбиение на дорожки, калибровка по маркеру, оценка размеров и флаги качества.

Зависимости: numpy, scipy, pillow, matplotlib, onnxruntime.
Модель: GelGenie Universal FineTune May-2024 (Aquilina et al., Nat Commun 2025, Apache-2.0),
сконвертирована в fp16 (вход/выход float32).

Пример:
  python analyze_gel.py gel.jpg --ladder-lanes 1 --ladder neb_1kb_plus \
      --expected "2=650,3=650,4=650" --negative 5 --out results/
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(HERE)

def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# работает и как Claude Skill (scripts/ + model/), и как файлы в одной папке (ChatGPT: /mnt/data)
DEFAULT_MODEL = _first_existing(os.path.join(SKILL_DIR, "model", "gelgenie_unet_fp16.onnx"),
                                os.path.join(HERE, "gelgenie_unet_fp16.onnx"),
                                os.path.join(HERE, "model", "gelgenie_unet_fp16.onnx"))
LADDERS_FILE = _first_existing(os.path.join(SKILL_DIR, "ladders.json"), os.path.join(HERE, "ladders.json"))
SMEAR_THR = 0.15
FIT_DEG = 3  # степень полинома log10(bp) ~ положение при подборе маркера
SKIP_PEN = 0.002  # штраф за каждую пропущенную внутреннюю полосу маркера
W_POLY = 1.0  # вес остатка полиномиальной аппроксимации
W_SMOOTH = 0.1  # вес локальной гладкости шага миграции
EXCLUDE_MERGED_FROM_INTERP = False


# ----------------------------------------------------------------------------- I/O
def load_ladders():
    with open(LADDERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_image(path):
    """Читает изображение -> (raw float32 в исходной шкале, dmax — предел шкалы, info).
    Учитывает EXIF-поворот (фото с телефона), палитровые/LA/CMYK/16-бит изображения и битность камеры;
    для цветных фото отдельно отмечает пиксели, упёршиеся в предел хотя бы одного канала (пересвет)."""
    from PIL import ImageOps
    im = Image.open(path)
    info = {"mode": im.mode}
    try:
        exif_orient = im.getexif().get(0x0112, 1)
    except Exception:
        exif_orient = 1
    if exif_orient not in (1, None):
        im = ImageOps.exif_transpose(im)
        info["exif_rotated"] = int(exif_orient)
    if im.mode in ("P", "PA", "LA", "CMYK", "YCbCr", "HSV", "LAB", "1", "RGBa", "La"):
        im = im.convert("RGBA" if im.mode in ("P", "PA") else "RGB" if im.mode not in ("LA", "La", "1") else "L")
    arr = np.array(im)
    clipped = None
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        cmax = 255.0 if arr.dtype == np.uint8 else 65535.0 if arr.dtype == np.uint16 else float(rgb.max() or 1)
        clipped = (rgb >= 0.995 * cmax).any(axis=2)
        # как cv2.COLOR_RGB2GRAY в GelGenie
        g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        arr = np.round(g).astype(arr.dtype if arr.dtype in (np.uint8, np.uint16) else np.float64)
    if arr.dtype == np.uint8:
        dmax = 255.0
    elif arr.dtype == np.uint16:
        dmax = 65535.0
        mx = int(arr.max())
        # 12/14-битные камеры пишут 16-битный TIFF, но значения не выше 4095/16383
        for bits in (10, 12, 14):
            if mx <= 2 ** bits - 1 and mx > 2 ** (bits - 2):
                dmax = float(2 ** bits - 1); info["bit_depth_detected"] = bits
                break
    else:  # int32 / float tif
        arr = arr.astype(np.float64)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-9) * 65535
        arr = arr.astype(np.uint16)
        dmax = 65535.0
    raw = arr.astype(np.float32)
    info["clipped_color_mask"] = clipped
    return raw, dmax, info


def detect_inverted(raw, dmax):
    """True, если полосы тёмные на светлом фоне (сканы/негативы)."""
    med = np.median(raw) / dmax
    return med > 0.5


# ------------------------------------------------------------------------ model
def run_model(img01, model_path, max_side=2048):
    import onnxruntime as ort
    H, W = img01.shape
    scale = 1.0
    x = img01
    if max(H, W) > max_side:  # большие сканы уменьшаем для скорости/памяти
        scale = max_side / max(H, W)
        x = np.array(Image.fromarray(img01).resize((int(W * scale), int(H * scale)), Image.BILINEAR))
    h, w = x.shape
    nh, nw = 32 * (h // 32 + 1), 32 * (w // 32 + 1)  # паддинг как в GelGenie
    t, l = (nh - h) // 2, (nw - w) // 2
    xp = np.pad(x, ((t, nh - h - t), (l, nw - w - l)), mode="constant").astype(np.float32)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: xp[None, None]})[0][0]
    mask = out.argmax(0)[t:t + h, l:l + w].astype(bool)
    if scale != 1.0:
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    return mask


# ------------------------------------------------------------------------ lanes
def find_runs(b):
    runs, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(b) - 1))
    return runs


def detect_lanes(mask, min_area):
    lab, n = ndi.label(mask)
    if n == 0:
        return [], {}
    sl = ndi.find_objects(lab)
    areas = ndi.sum(mask, lab, range(1, n + 1))
    keep = [i for i in range(n) if areas[i] >= min_area]
    if not keep:
        return [], {}
    widths = np.array([sl[i][1].stop - sl[i][1].start for i in keep])
    bw = float(np.median(widths))
    clean = np.isin(lab, [i + 1 for i in keep])
    prof = ndi.gaussian_filter1d(clean.sum(0).astype(float), 2)
    thr = max(prof.max() * 0.02, 0.5)
    runs = [r for r in find_runs(prof > thr) if (r[1] - r[0] + 1) >= 0.3 * bw]
    # слишком широкие "дорожки" = несколько слипшихся, делим по провалам профиля
    lanes = []
    for a, b in runs:
        width = b - a + 1
        k = int(round(width / (bw * 1.15)))
        if k <= 1:
            lanes.append((a, b)); continue
        seg = prof[a:b + 1]
        cuts = []
        for j in range(1, k):
            c = int(j * width / k)
            lo, hi = max(1, c - int(bw * 0.35)), min(width - 2, c + int(bw * 0.35))
            cuts.append(a + lo + int(np.argmin(seg[lo:hi + 1])))
        edges = [a] + cuts + [b]
        for i in range(len(edges) - 1):
            lanes.append((edges[i] if i == 0 else edges[i] + 1, edges[i + 1]))
    centers = [0.5 * (a + b) for a, b in lanes]
    info = {"band_width_px": bw, "n_detected": len(lanes)}
    # вставляем пустые дорожки, если шаг между соседями ~2x
    if len(centers) >= 3:
        pitch = float(np.median(np.diff(centers)))
        info["pitch_px"] = pitch
        out = [lanes[0]]
        for prev, cur in zip(lanes[:-1], lanes[1:]):
            gap = 0.5 * (cur[0] + cur[1]) - 0.5 * (prev[0] + prev[1])
            m = int(round(gap / pitch)) - 1
            for j in range(1, m + 1):
                c = 0.5 * (prev[0] + prev[1]) + j * gap / (m + 1)
                out.append((int(c - bw / 2), int(c + bw / 2), "empty"))
            out.append(cur)
        lanes = out
    res = []
    for ln in lanes:
        res.append({"x0": int(ln[0]), "x1": int(ln[1]), "empty_inferred": len(ln) == 3})
    return res, info


def lane_bounds(lanes, W):
    """Границы колонок для каждой дорожки: середины между соседями."""
    c = [0.5 * (l["x0"] + l["x1"]) for l in lanes]
    b = []
    for i in range(len(c)):
        left = 0 if i == 0 else int(0.5 * (c[i - 1] + c[i])) + 1
        right = W - 1 if i == len(c) - 1 else int(0.5 * (c[i] + c[i + 1]))
        # не уходим дальше полутора ширин полосы от центра
        hw = 0.75 * (lanes[i]["x1"] - lanes[i]["x0"] + 1)
        left = max(left, int(c[i] - hw)); right = min(right, int(c[i] + hw))
        b.append((left, right))
    return b


# ------------------------------------------------------------------------ bands
def measure_bands(mask, sig, dmax_raw, raw, bounds, min_area, min_width=0):
    """sig: сигнал, где полосы светлые. raw: исходные значения для проверки пересвета."""
    lanes_out = []
    for li, (x0, x1) in enumerate(bounds):
        sub = mask[:, x0:x1 + 1]
        lab, n = ndi.label(sub)
        sig_sub = sig[:, x0:x1 + 1]
        bg = float(np.median(sig_sub[~ndi.binary_dilation(sub, iterations=3)])) if (~sub).any() else 0.0
        bands = []
        for k in range(1, n + 1):
            m = lab == k
            area = int(m.sum())
            ys, xs = np.nonzero(m)
            # отбрасываем мелочь и обрезки соседних полос, попавшие на границу дорожки
            if area < min_area or (xs.max() - xs.min() + 1) < min_width:
                continue
            vals = sig_sub[m]
            w = np.clip(vals - bg, 0, None)
            y_c = float((ys * w).sum() / w.sum()) if w.sum() > 0 else float(ys.mean())
            raw_vals = raw[:, x0:x1 + 1][m]
            sat = float((raw_vals >= 0.995 * dmax_raw).mean())
            bands.append({
                "y": round(y_c, 1), "y_top": int(ys.min()), "y_bottom": int(ys.max()),
                "x_left": int(xs.min() + x0), "x_right": int(xs.max() + x0),
                "height_px": int(ys.max() - ys.min() + 1), "width_px": int(xs.max() - xs.min() + 1),
                "area_px": area, "volume": float(w.sum()), "saturated_frac": round(sat, 3),
            })
        bands.sort(key=lambda b: b["y"])
        tot = sum(b["volume"] for b in bands) or 1.0
        for b in bands:
            b["rel_intensity_pct"] = round(100 * b["volume"] / tot, 1)
        lanes_out.append({"lane": li + 1, "x0": x0, "x1": x1, "x_center": int((x0 + x1) / 2), "background": round(bg, 2),
                          "bands": bands})
    return lanes_out


def smear_index(lanes, mask, sig):
    """Сигнал между/под полосами относительно соседних межлинейных промежутков на тех же высотах
    (компенсирует градиент фона), нормированный на яркость полос. ~0 у чистой дорожки, >0.15 у смазанной."""
    H, W = sig.shape
    for L in lanes:
        bands = L["bands"]
        if not bands:
            L["smear_index"] = 0.0; continue
        xl = min(b["x_left"] for b in bands); xr = max(b["x_right"] for b in bands)
        bw = xr - xl + 1
        c0, c1 = xl + int(0.3 * bw), xr - int(0.3 * bw)
        prof = np.median(sig[:, c0:c1 + 1], axis=1)
        strips = []
        for g0, g1 in ((xl - 12, xl - 3), (xr + 3, xr + 12)):
            g0, g1 = max(0, g0), min(W - 1, g1)
            if g1 > g0:
                st = sig[:, g0:g1 + 1].astype(float).copy(); st[mask[:, g0:g1 + 1]] = np.nan
                strips.append(st)
        if not strips:
            L["smear_index"] = 0.0; continue
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = np.nanmedian(np.concatenate(strips, 1), axis=1)
        y0 = min(b["y_top"] for b in bands)
        y1 = min(H - 1, max(b["y_bottom"] for b in bands) + int(0.15 * H))
        rows = np.zeros(H, bool); rows[y0:y1 + 1] = True
        for b in bands:
            rows[max(0, b["y_top"] - 3):b["y_bottom"] + 4] = False
        rows &= ~np.isnan(ref)
        peaks = [prof[int(round(b["y"]))] - ref[int(round(b["y"]))] for b in bands if not np.isnan(ref[int(round(b["y"]))])]
        peak = float(np.median(peaks)) if peaks else 0.0
        if rows.sum() < 5 or peak <= 0:
            L["smear_index"] = 0.0; continue
        L["smear_index"] = round(max(0.0, float(np.median(prof[rows] - ref[rows])) / peak), 3)
    return lanes


def tall_band_indices(bands, ratio=1.6):
    """Индексы полос, которые заметно выше соседей (по высоте) — вероятно, две слившиеся полосы.
    Сравнение с ближайшими соседями, т.к. мелкие фрагменты внизу геля закономерно шире."""
    h = np.array([b["height_px"] for b in bands], float)
    out = set()
    for i in range(len(h)):
        nb = np.r_[h[max(0, i - 2):i], h[i + 1:i + 3]]
        if len(nb) >= 2 and h[i] > ratio * np.median(nb) and h[i] >= 10:
            out.add(i)
    return out


# ------------------------------------------------------------------- calibration
def _score(ys, logs_sel, skips, logv=None, logw_sel=None, lam=0.0):
    A = np.stack([ys ** k for k in range(FIT_DEG + 1)], 1)
    P = np.linalg.pinv(A)
    coef = logs_sel @ P.T
    res = logs_sel - coef @ A.T
    rms = np.sqrt((res ** 2).mean(1))
    score = W_POLY * rms + SKIP_PEN * skips
    if W_SMOOTH > 0 and len(ys) >= 4:
        # локальная гладкость: «скорость» миграции (px на единицу log10 bp) между соседними полосами
        # должна меняться плавно; сдвиг подписей на полосу ломает её около неравномерностей маркера
        dy = np.diff(ys)
        dl = -np.diff(logs_sel, axis=1)
        lsl = np.log(np.maximum(dy, 1e-9) / np.maximum(dl, 1e-9))
        d2 = np.diff(lsl, axis=1)
        score = score + W_SMOOTH * np.sqrt((d2 ** 2).mean(1))
    if logv is not None and logw_sel is not None and lam > 0 and len(logv) >= 4:
        # ЛОКАЛЬНОЕ согласие яркостей: насколько полоса ярче соседей (log-отношение) против того,
        # насколько полоса маркера массивнее соседей по паспорту. Локальность убирает общий тренд
        # «крупные полосы ярче/тусклее», который зависит от геля, красителя и съёмки.
        d = logv[1:-1] - 0.5 * (logv[:-2] + logv[2:])
        e = logw_sel[:, 1:-1] - 0.5 * (logw_sel[:, :-2] + logw_sel[:, 2:])
        dc = d - d.mean(); ec = e - e.mean(1, keepdims=True)
        den = np.sqrt((dc ** 2).sum() * (ec ** 2).sum(1))
        r = np.where(den > 0, (ec @ dc) / np.where(den > 0, den, 1), 0.0)
        score = score + lam * (1 - r) / 2
    return score, rms


def match_ladder(band_ys, sizes, H, volumes=None, weights=None, lam=0.08, max_combos=400000, first_bp=None):
    """Сопоставляет обнаруженные полосы маркера с паспортными размерами.
    Перебор монотонных соответствий, критерий: остаток полиномиальной (FIT_DEG) аппроксимации log10(bp)~y
    + штраф за пропуск внутренних полос маркера
    + штраф за негладкость локального шага миграции
    + штраф за несогласие яркостей (опорные/более массивные полосы маркера должны быть ярче)."""
    ys = np.asarray(band_ys, float) / H
    order = np.argsort(sizes)[::-1]
    sizes = np.asarray(sizes, float)[order]
    logs = np.log10(sizes)
    logw = np.log(np.asarray(weights, float)[order]) if weights is not None else None
    logv = np.log(np.maximum(np.asarray(volumes, float), 1e-6)) if (volumes is not None and logw is not None) else None
    nd, nl = len(ys), len(sizes)
    if nd < 3:
        return None
    if nd <= nl:
        n_comb = 1
        for i in range(nd):
            n_comb = n_comb * (nl - i) // (i + 1)
        if n_comb > max_combos:
            return None
        combos = np.array(list(itertools.combinations(range(nl), nd)))
        if first_bp is not None:  # пользователь/агент указал размер самой верхней обнаруженной полосы маркера
            j = int(np.argmin(np.abs(np.log(sizes / first_bp))))
            combos = combos[combos[:, 0] == j]
            if len(combos) == 0:
                return None
        skips = (combos[:, -1] - combos[:, 0] + 1 - nd).astype(float)
        score, rms = _score(ys, logs[combos], skips, logv, logw[combos] if logv is not None else None, lam)
        i = int(np.argmin(score))
        det_idx = list(range(nd)); lad_idx = list(combos[i])
        best_rms = float(rms[i])
    else:  # лишние полосы (артефакты/дублеты) — выбираем подмножество обнаруженных
        best = None
        n_comb = 1
        for i in range(nl):
            n_comb = n_comb * (nd - i) // (i + 1)
        if n_comb > 50000:
            return None
        for det in itertools.combinations(range(nd), nl):
            det = list(det)
            s, r = _score(ys[det], logs[None, :], np.array([0.0]),
                          logv[det] if logv is not None else None, logw[None, :] if logv is not None else None, lam)
            s = float(s[0]) + 0.006 * (nd - nl)
            if best is None or s < best[0]:
                best = (s, float(r[0]), det)
        det_idx = best[2]; lad_idx = list(range(nl)); best_rms = best[1]
    pairs = [(float(band_ys[d]), float(sizes[l])) for d, l in zip(det_idx, lad_idx)]
    # ошибка калибровки: leave-one-out по внутренним точкам, % от размера
    yy = np.array([p[0] for p in pairs]); ll = np.log10([p[1] for p in pairs])
    loo = []
    for j in range(1, len(pairs) - 1):
        m = np.ones(len(pairs), bool); m[j] = False
        pred = np.interp(yy[j], yy[m], ll[m])
        loo.append(abs(10 ** pred / 10 ** ll[j] - 1) * 100)
    out = {"pairs": pairs, "fit_rms_log10": round(best_rms, 4),
           "loo_median_err_pct": round(float(np.median(loo)), 1) if loo else None,
           "loo_max_err_pct": round(float(np.max(loo)), 1) if loo else None,
           "unmatched_detected": [i for i in range(nd) if i not in det_idx],
           "missing_ladder_bp": [float(sizes[i]) for i in range(nl) if i not in lad_idx]}
    if volumes is not None and len(det_idx) >= 3:
        # какие полосы маркера оказались ЛОКАЛЬНО самыми яркими (ярче соседей) — для проверки агентом:
        # должны совпасть с опорными (bright_bp) полосами маркера
        v = np.log(np.maximum([volumes[d] for d in det_idx], 1e-6))
        contrast = [v[i] - 0.5 * ((v[i - 1] if i > 0 else v[i + 1]) + (v[i + 1] if i < len(v) - 1 else v[i - 1]))
                    for i in range(len(v))]
        top = np.argsort(contrast)[::-1][:3]
        out["locally_brightest_assigned_bp"] = [float(sizes[lad_idx[i]]) for i in top]
    return out


def predict_bp(y, cal):
    pairs = cal.get("interp_pairs") or cal["pairs"]
    yy = np.array([p[0] for p in pairs]); ll = np.log10([p[1] for p in pairs])
    o = np.argsort(yy); yy, ll = yy[o], ll[o]
    if yy[0] <= y <= yy[-1]:
        return float(10 ** np.interp(y, yy, ll)), False
    # вне маркера — линейная экстраполяция по 3 крайним полосам (глобальный полином у краёв «загибается»)
    sel = slice(0, 3) if y < yy[0] else slice(-3, None)
    c = np.polyfit(yy[sel], ll[sel], 1)
    return float(10 ** np.polyval(c, y)), True


# ------------------------------------------------------------------------ mass
def mass_calibration(ladder_bands, sizes, masses, total_ng, dmax_sat=0.02):
    """Калибровка «объём сигнала -> нг» по полосам маркера с известной массой.
    Модель: volume = k(bp) * ng, где log k линейно зависит от log bp (учитывает, что мелкие/крупные
    полосы на конкретном геле светятся по-разному в пересчёте на нг)."""
    mass_of = {float(sz): m * total_ng / 1000.0 for sz, m in zip(sizes, masses)}
    pts = [(b["ladder_bp"], mass_of[b["ladder_bp"]], b["volume"]) for b in ladder_bands
           if b.get("ladder_bp") in mass_of and b["saturated_frac"] <= dmax_sat and b["volume"] > 0]
    if len(pts) < 4:
        return None
    bp, ng, vol = (np.array(x, float) for x in zip(*pts))
    lr = np.log(vol / ng); lb = np.log10(bp)
    c = np.polyfit(lb, lr, 1)
    resid = lr - np.polyval(c, lb)
    # одна итерация отбрасывания выбросов (слившиеся/обрезанные полосы)
    keep = np.abs(resid) < max(2.5 * np.median(np.abs(resid)), 0.15)
    if keep.sum() >= 4:
        c = np.polyfit(lb[keep], lr[keep], 1); resid = lr - np.polyval(c, lb)
    loo = []
    for j in range(len(bp)):
        m = np.ones(len(bp), bool); m[j] = False
        cj = np.polyfit(lb[m], lr[m], 1)
        loo.append(abs(vol[j] / np.exp(np.polyval(cj, lb[j])) / ng[j] - 1) * 100)
    return {"coef": [float(x) for x in c], "n_bands": int(len(bp)), "ng_range": [float(ng.min()), float(ng.max())],
            "vol_range": [float(vol.min()), float(vol.max())],
            "loo_median_err_pct": round(float(np.median(loo)), 1), "loo_max_err_pct": round(float(np.max(loo)), 1)}


def predict_ng(volume, bp, mc):
    k = np.exp(np.polyval(mc["coef"], np.log10(bp)))
    return float(volume / k)


# ------------------------------------------------------------------------- flags
def parse_expected(s):
    """'2=650,3=650/1200' (также допускает пробелы, 'bp'/'п.н.', ';' и ':' вместо '=')."""
    import re
    out = {}
    if not s:
        return out
    for part in re.split(r"[;,]\s*(?=\d+\s*[=:])", s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*[=:]\s*(.+)$", part)
        if not m:
            sys.exit(f"Не понял --expected '{part}'. Формат: '2=650,3=650/1200'")
        sizes = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(2))]
        if not sizes:
            sys.exit(f"Не понял --expected '{part}': нет размеров")
        out[int(m.group(1))] = sizes
    return out


def build_flags(lanes, ladder_lanes, negatives, expected, tol, cal_ok):
    for L in lanes:
        f = []
        n = L["lane"]
        if n in ladder_lanes:
            L["role"] = "ladder"
        elif n in negatives:
            L["role"] = "negative_control"
        else:
            L["role"] = "sample"
        bands = L["bands"]
        if not bands:
            f.append("нет полос" + (" (дорожка достроена по шагу лунок)" if L.get("empty_inferred") else ""))
        if L["role"] == "negative_control" and bands:
            f.append("СИГНАЛ В ОТРИЦАТЕЛЬНОМ КОНТРОЛЕ: возможна контаминация")
        if any(b["saturated_frac"] > 0.05 for b in bands):
            f.append("пересвет: часть полос насыщена, количественная оценка занижена")
        if L["smear_index"] > SMEAR_THR and bands:
            f.append("много сигнала вне полос: смазанность/деградация/перегруз")
        if bands:
            med_h = np.median([b["height_px"] for b in bands])
            if any(b["height_px"] > 3 * max(med_h, 4) for b in bands):
                f.append("есть вытянутая полоса (смазанность или слившиеся полосы)")
        if L["role"] != "ladder" and cal_ok:
            if any(b.get("bp") is not None and b["bp"] < 120 for b in bands):
                f.append("полоса <120 п.н.: возможны праймер-димеры")
            if any(b.get("extrapolated") for b in bands):
                f.append("есть полосы вне диапазона маркера (размер экстраполирован)")
        if n in expected and cal_ok:
            hits, extra = [], []
            for b in bands:
                if b.get("bp") is None:
                    continue
                m = [e for e in expected[n] if abs(b["bp"] / e - 1) <= tol]
                (hits if m else extra).append(b)
            found = sorted({e for e in expected[n] for b in hits if abs(b["bp"] / e - 1) <= tol})
            L["expected_bp"] = expected[n]
            L["expected_found"] = found
            if len(found) == len(expected[n]):
                f.append("ожидаемый продукт найден")
            else:
                miss = [e for e in expected[n] if e not in found]
                f.append(f"ожидаемый продукт НЕ найден: {', '.join(str(int(m)) for m in miss)} п.н.")
            if extra:
                f.append("дополнительные полосы: " + ", ".join(f"~{int(round(b['bp']))} п.н." for b in extra))
        L["flags"] = f
    return lanes


# ---------------------------------------------------------------------- plotting
def plot_overlay(sig, mask, lanes, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = sig.shape
    fig_w = 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * H / W * 0.92 + 0.6), dpi=110)
    lo, hi = np.percentile(sig, 1), np.percentile(sig, 99.7)
    ax.imshow(np.clip((sig - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
    edge = mask ^ ndi.binary_erosion(mask, iterations=1)
    ov = np.zeros((H, W, 4)); ov[edge] = [1.0, 0.75, 0.0, 1.0]
    ax.imshow(ov)
    for L in lanes:
        cx = 0.5 * (L["x0"] + L["x1"])
        col = {"ladder": "#4FC3F7", "negative_control": "#EF5350"}.get(L["role"], "#FFFFFF")
        ax.text(cx, 8, str(L["lane"]), color=col, ha="center", va="top", fontsize=10, weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, lw=0))
        ax.axvline(L["x0"], color="#ffffff", lw=0.3, alpha=0.25)
        for b in L["bands"]:
            if b.get("bp") is not None and L["role"] != "ladder":
                lab = f"{int(round(b['bp']))}" + ("?" if b.get("extrapolated") else "")
                ax.text(b["x_right"] + 2, b["y"], lab, color="#7CFC00", fontsize=7, va="center",
                        bbox=dict(facecolor="black", alpha=0.55, pad=0.5, lw=0))
            elif b.get("ladder_bp") is not None:
                ax.text(b["x_left"] - 2, b["y"], f"{int(b['ladder_bp'])}", color="#4FC3F7", fontsize=6,
                        ha="right", va="center", bbox=dict(facecolor="black", alpha=0.55, pad=0.5, lw=0))
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(0, W, 100)); ax.tick_params(axis="x", labelsize=7)
    ax.set_yticks(np.arange(0, H, 100)); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_calibration(cals, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=110)
    for lane, cal in cals.items():
        y = [p[0] for p in cal["pairs"]]; s = [p[1] for p in cal["pairs"]]
        ax.semilogy(y, s, "o-", ms=4, label=f"маркер, дорожка {lane}")
    ax.set_xlabel("положение полосы, px (сверху вниз)"); ax.set_ylabel("размер, п.н.")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


# -------------------------------------------------------------------------- main
def analyze(args):
    os.makedirs(args.out, exist_ok=True)
    raw, dmax, img_info = load_image(args.image)
    clipped_rgb = img_info.pop("clipped_color_mask", None)
    H, W = raw.shape
    inv = detect_inverted(raw, dmax) if args.invert == "auto" else args.invert == "yes"
    sig = (dmax - raw) if inv else raw.copy()
    img01 = sig / dmax
    if args.norm == "minmax":
        img01 = (sig - sig.min()) / max(sig.max() - sig.min(), 1e-6)
    elif args.norm == "percentile":
        lo, hi = np.percentile(sig, 0.1), np.percentile(sig, 99.9)
        img01 = (np.clip(sig, lo, hi) - lo) / max(hi - lo, 1e-6)
    mask = run_model(img01.astype(np.float32), args.model)
    min_area = max(12, int(H * W * 2e-5))
    lanes_geo, lane_info = detect_lanes(mask, min_area)
    if args.lane_centers:  # ручное задание дорожек (x центров лунок, px) — после проверки по overlay
        cs = sorted(float(c) for c in args.lane_centers.split(","))
        pitch = float(np.median(np.diff(cs))) if len(cs) > 1 else lane_info.get("band_width_px", 40) * 1.5
        hw = 0.5 * min(lane_info.get("band_width_px", pitch * 0.8), pitch * 0.9)
        lanes_geo = [{"x0": int(c - hw), "x1": int(c + hw), "empty_inferred": False} for c in cs]
        lane_info["manual_centers"] = cs
    if (args.add_empty_left or args.add_empty_right) and not lanes_geo:
        sys.exit("Полосы не найдены — нельзя достроить пустые лунки; задайте --lane-centers вручную")
    if "pitch_px" in lane_info:
        pitch = lane_info["pitch_px"]
    elif len(lanes_geo) == 2:
        pitch = 0.5 * (lanes_geo[1]["x0"] + lanes_geo[1]["x1"]) - 0.5 * (lanes_geo[0]["x0"] + lanes_geo[0]["x1"])
    else:
        pitch = 1.6 * lane_info.get("band_width_px", 40)
    for _ in range(args.add_empty_left):
        f = lanes_geo[0]
        lanes_geo.insert(0, {"x0": int(f["x0"] - pitch), "x1": int(f["x1"] - pitch), "empty_inferred": True})
    for _ in range(args.add_empty_right):
        f = lanes_geo[-1]
        lanes_geo.append({"x0": int(f["x0"] + pitch), "x1": int(f["x1"] + pitch), "empty_inferred": True})
    n_before = len(lanes_geo)
    lanes_geo = [l for l in lanes_geo if l["x1"] > 0 and l["x0"] < W]
    early_warnings = []
    if len(lanes_geo) < n_before:
        early_warnings.append(f"{n_before - len(lanes_geo)} достроенных дорожек вышли за край снимка и отброшены — "
                              "нумерация могла сместиться, проверьте --add-empty-left/right")
    if img_info.get("exif_rotated"):
        early_warnings.append("Снимок повёрнут по EXIF (фото с телефона) перед анализом")
    if img_info.get("bit_depth_detected"):
        early_warnings.append(f"16-битный файл с {img_info['bit_depth_detected']}-битными данными: шкала приведена к 0–{int(dmax)}")
    bounds = lane_bounds(lanes_geo, W)
    # sig уже "полосы светлые": пересвет = sig у предела шкалы (для негатива это чёрные пиксели исходника);
    # у цветных фото — ещё и упор любого из каналов RGB в предел
    sat_src = sig.copy()
    if clipped_rgb is not None and not inv:
        sat_src[clipped_rgb] = dmax
    lanes = measure_bands(mask, sig, dmax, sat_src, bounds, min_area, 0.4 * lane_info.get("band_width_px", 0))
    lanes = smear_index(lanes, mask, sig)
    for L, g in zip(lanes, lanes_geo):
        L["empty_inferred"] = g["empty_inferred"]

    ladders = load_ladders()
    ladder_lanes = [int(x) for x in args.ladder_lanes.split(",")] if args.ladder_lanes else []
    negatives = [int(x) for x in args.negative.split(",")] if args.negative else []
    expected = parse_expected(args.expected)
    sizes, weights, bright_set = None, None, set()
    if args.ladder_sizes:
        sizes = [float(x) for x in args.ladder_sizes.split(",")]
    elif args.ladder:
        if args.ladder not in ladders:
            sys.exit(f"Неизвестный маркер {args.ladder}. Доступны: {', '.join(ladders)}")
        sizes = ladders[args.ladder]["sizes_bp"]
        lad = ladders[args.ladder]
        bright_set = {float(x) for x in lad.get("bright_bp", [])}
        if lad.get("mass_ng_per_ug"):
            weights = lad["mass_ng_per_ug"]
        elif lad.get("bright_bp"):
            weights = [2.0 if x in lad["bright_bp"] else 1.0 for x in sizes]

    cals, warnings = {}, list(early_warnings)
    if sizes and ladder_lanes:
        for ln in ladder_lanes:
            if ln < 1 or ln > len(lanes):
                warnings.append(f"Дорожка маркера {ln} не найдена (обнаружено дорожек: {len(lanes)})"); continue
            lb = lanes[ln - 1]["bands"]
            ys = [b["y"] for b in lb]
            vols = [b["volume"] for b in lb]
            cal = match_ladder(ys, sizes, H, volumes=vols, weights=weights, lam=args.bright_weight,
                               first_bp=getattr(args, "ladder_first_bp", None))
            if cal is None:
                warnings.append(f"Не удалось откалибровать маркер в дорожке {ln} (полос: {len(ys)})"); continue
            # слившиеся полосы маркера (заметно выше соседей): их центр лежит между двумя размерами —
            # в сопоставлении участвуют, но из интерполяции размеров исключаются
            # опорные полосы маркера (bright_bp) толще соседей закономерно — их слившимися не считаем
            tall = tall_band_indices(lb)
            ladder_of = {p[0]: p[1] for p in cal["pairs"]}
            tall_y = {lb[i]["y"] for i in tall if ladder_of.get(lb[i]["y"]) not in bright_set}
            cal["merged_bands_bp"] = [p[1] for p in cal["pairs"] if p[0] in tall_y]
            interp = [p for p in cal["pairs"] if p[0] not in tall_y]
            cal["interp_pairs"] = interp if (len(interp) >= 3 and EXCLUDE_MERGED_FROM_INTERP) else cal["pairs"]
            if cal["merged_bands_bp"]:
                warnings.append(f"В маркере (дорожка {ln}) полосы {', '.join(str(int(x)) for x in cal['merged_bands_bp'])} "
                                "похожи на слившиеся (заметно шире соседних): размеры рядом с ними менее точны")
            if len(ys) < 6:
                warnings.append(f"В маркере (дорожка {ln}) всего {len(ys)} полос — сопоставление с паспортом "
                                "неоднозначно; проверьте подписи на overlay или задайте --ladder-first-bp")
            if bright_set and cal.get("locally_brightest_assigned_bp"):
                present = bright_set & {p[1] for p in cal["pairs"]}
                need = min(2, len(present))
                if need and len(present & set(cal["locally_brightest_assigned_bp"])) < need:
                    warnings.append(f"Опорные полосы маркера (дорожка {ln}) не совпали с самыми яркими "
                                    f"({', '.join(str(int(x)) for x in cal['locally_brightest_assigned_bp'])}): "
                                    "возможен сдвиг подписей на полосу или другая версия маркера")
            cals[ln] = cal
            for b in lanes[ln - 1]["bands"]:
                for y, s in cal["pairs"]:
                    if abs(b["y"] - y) < 1e-6:
                        b["ladder_bp"] = s
    cal_ok = bool(cals)
    if cal_ok:
        centers = {ln: 0.5 * (lanes[ln - 1]["x0"] + lanes[ln - 1]["x1"]) for ln in cals}
        for L in lanes:
            if L["lane"] in cals:
                continue
            cx = 0.5 * (L["x0"] + L["x1"])
            ks = sorted(cals, key=lambda k: abs(centers[k] - cx))[:2]
            for b in L["bands"]:
                preds = [predict_bp(b["y"], cals[k]) for k in ks]
                if len(ks) == 2 and centers[ks[0]] != centers[ks[1]]:
                    # линейная интерполяция по x между двумя маркерами (поправка на "улыбку")
                    w0 = abs(centers[ks[1]] - cx) / (abs(centers[ks[0]] - cx) + abs(centers[ks[1]] - cx))
                    bp = 10 ** (w0 * np.log10(preds[0][0]) + (1 - w0) * np.log10(preds[1][0]))
                else:
                    bp = preds[0][0]
                b["bp"] = round(float(bp), 0)
                b["extrapolated"] = any(p[1] for p in preds)
        for ln, cal in cals.items():
            if cal["loo_median_err_pct"] is not None and cal["loo_median_err_pct"] > 8:
                warnings.append(f"Калибровка по дорожке {ln} ненадёжна (медианная ошибка {cal['loo_median_err_pct']}%)."
                                " Проверьте тип маркера и соответствие полос на overlay.png")
            if cal["unmatched_detected"]:
                warnings.append(f"В маркере (дорожка {ln}) {len(cal['unmatched_detected'])} лишних полос не сопоставлено")
    mcal = {}
    if cal_ok and args.ladder_ng:
        lad = ladders.get(args.ladder, {}) if args.ladder else {}
        if not lad.get("mass_ng_per_ug"):
            warnings.append("Для этого маркера в ladders.json нет масс полос — оценка нг невозможна")
        else:
            for ln in cals:
                mc = mass_calibration(lanes[ln - 1]["bands"], lad["sizes_bp"], lad["mass_ng_per_ug"], args.ladder_ng)
                if mc:
                    mcal[ln] = mc
            if mcal:
                for L in lanes:
                    if L["lane"] in cals:
                        continue
                    cx = 0.5 * (L["x0"] + L["x1"])
                    k = min(mcal, key=lambda q: abs(0.5 * (lanes[q - 1]["x0"] + lanes[q - 1]["x1"]) - cx))
                    mc = mcal[k]
                    for b in L["bands"]:
                        if not b.get("bp"):
                            continue
                        ng = predict_ng(b["volume"], b["bp"], mc)
                        b["ng"] = round(ng, 1)
                        note = []
                        if b["saturated_frac"] > 0.02:
                            note.append("пересвет: нижняя граница")
                        if b["volume"] > 1.5 * mc["vol_range"][1]:
                            note.append("ярче всех полос маркера: экстраполяция")
                        if b["volume"] < 0.5 * mc["vol_range"][0]:
                            note.append("тусклее всех полос маркера: экстраполяция")
                        if b.get("extrapolated"):
                            note.append("размер вне маркера")
                        if note:
                            b["ng_note"] = "; ".join(note)
                        if args.sample_ul:
                            b["ng_per_ul_loaded"] = round(ng / args.sample_ul, 2)
                    L["lane_total_ng"] = round(sum(b.get("ng", 0) for b in L["bands"]), 1)
            else:
                warnings.append("Не удалось построить калибровку по массе (мало ненасыщенных полос маркера)")
    lanes = build_flags(lanes, ladder_lanes, negatives, expected, args.tolerance, cal_ok)

    base = os.path.splitext(os.path.basename(args.image))[0]
    ov_png = os.path.join(args.out, f"{base}_overlay.png")
    plot_overlay(sig, mask, lanes, ov_png, title=f"{base}: GelGenie сегментация (жёлтые контуры), размеры в п.н.")
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, f"{base}_mask.png"))
    if cals:
        plot_calibration(cals, os.path.join(args.out, f"{base}_calibration.png"))

    result = {
        "image": os.path.basename(args.image), "size_px": [W, H], "inverted": inv,
        "model": os.path.basename(args.model), "lanes_detected": len(lanes), "lane_detection": lane_info,
        "ladder": args.ladder or ("custom" if sizes else None), "ladder_lanes": ladder_lanes,
        "calibration": {str(k): v for k, v in cals.items()},
        "mass_calibration": {str(k): v for k, v in mcal.items()}, "ladder_ng_loaded": args.ladder_ng,
        "warnings": warnings, "lanes": lanes,
    }
    with open(os.path.join(args.out, f"{base}_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # краткая таблица в markdown
    lines = [f"# Гель {base}", "",
             f"Дорожек: {len(lanes)}; инверсия: {'да' if inv else 'нет'}; маркер: {result['ladder']} в дорожках {ladder_lanes or '—'}"]
    for ln, cal in cals.items():
        lines.append(f"Калибровка (дорожка {ln}): сопоставлено {len(cal['pairs'])} полос, "
                     f"медианная ошибка LOO {cal['loo_median_err_pct']}%, макс {cal['loo_max_err_pct']}%")
    for ln, mc in mcal.items():
        lines.append(f"Калибровка по массе (дорожка {ln}, {args.ladder_ng} нг маркера): {mc['n_bands']} полос, "
                     f"{mc['ng_range'][0]:.0f}–{mc['ng_range'][1]:.0f} нг, ошибка LOO медиана {mc['loo_median_err_pct']}%, "
                     f"макс {mc['loo_max_err_pct']}%. Оценка нг полуколичественная.")
    for w in warnings:
        lines.append(f"ВНИМАНИЕ: {w}")
    lines += ["", "| Дорожка | Роль | Полосы, п.н. (отн. интенсивность % или ~нг) | Флаги |", "|---|---|---|---|"]
    for L in lanes:
        if L["role"] == "ladder":
            bands = f"{len(L['bands'])} полос маркера"
        else:
            bands = "; ".join((f"{int(b['bp'])}{'?' if b.get('extrapolated') else ''}" if b.get("bp") else f"y={b['y']:.0f}")
                              + (f" (~{b['ng']:.0f} нг{'*' if b.get('ng_note') else ''})" if b.get("ng") is not None
                                 else f" ({b['rel_intensity_pct']}%)") for b in L["bands"]) or "—"
        lines.append(f"| {L['lane']} | {L['role']} | {bands} | {'; '.join(L['flags']) or '—'} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out, f"{base}_report.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nФайлы: {ov_png}, *_results.json, *_report.md" + (", *_calibration.png" if cals else ""))
    return result


def main():
    p = argparse.ArgumentParser(description="GelGenie-агент: анализ фото геля")
    p.add_argument("image")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="gel_results")
    p.add_argument("--ladder", help="пресет маркера из ladders.json")
    p.add_argument("--ladder-sizes", help="свой маркер: размеры через запятую, п.н.")
    p.add_argument("--ladder-lanes", help="номера дорожек маркера, слева направо с 1, напр. 1 или 1,12")
    p.add_argument("--negative", help="дорожки отрицательного контроля, напр. 8")
    p.add_argument("--expected", help="ожидаемые продукты: '2=650,3=650/1200'")
    p.add_argument("--bright-weight", type=float, default=0.08,
                   help="вес согласия яркостей при сопоставлении маркера (0 = только по положению)")
    p.add_argument("--ladder-first-bp", type=float,
                   help="размер самой верхней обнаруженной полосы маркера (если автоматическое сопоставление ошиблось)")
    p.add_argument("--ladder-ng", type=float, help="сколько нг маркера внесено в лунку (для оценки массы ДНК в полосах)")
    p.add_argument("--sample-ul", type=float, help="объём образца в лунке, мкл (тогда добавится нг/мкл)")
    p.add_argument("--tolerance", type=float, default=0.10, help="допуск по размеру (доля), по умолчанию 0.10")
    p.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--norm", choices=["none", "minmax", "percentile"], default="none",
                   help="none = как в GelGenie по умолчанию; percentile помогает на тусклых фото")
    p.add_argument("--lane-centers", help="ручные x-координаты центров дорожек через запятую (px, см. шкалу на overlay)")
    p.add_argument("--add-empty-left", type=int, default=0, help="добавить N пустых дорожек слева")
    p.add_argument("--add-empty-right", type=int, default=0, help="добавить N пустых дорожек справа")
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
