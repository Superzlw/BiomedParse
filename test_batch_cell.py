#!/usr/bin/env python
"""BiomedParse v1 细胞分割批量评测（细胞核多边形标注的测试集，例如 WNS 的 QUPATH_WNS_FINAL）。

和 VPR_LISA_BASED/scripts/test_batch_cells.py（我们的模型）用同一套数据、同一个问题集、同一套打分：
  * 每张 tile 对训练用的 8 个细胞类都发 prompt（不管这类在不在 tile 里），概率 > --thr 为预测；
  * 8 张概率图合成一张带类别的细胞核图：概率最高、且超过阈值的类赢这个像素
    （= VPR 的 utils/instance_metrics.typed_from_logits，输入 logit = 概率 - 阈值）；
  * 细胞核实例指标 = VPR 的 utils/instance_metrics.HoverMeter.update_tile（直接从 --vpr-repo 加载同一个文件，
    预测按距离变换分水岭切成细胞核，margins 0 / 16 px）：binary Dice / AJI / PQ、每类 PQ+、mPQ+、mPQ(present)；
  * 每类像素 Dice：GT 里这类 >= --min-cell-px（200）的 tile 上算 mean / pooled / P / R；其余 tile 报误画率
    （这里 = 阈值以上的像素 > 0；我们的模型 = 给了 [SEG]）；
  * 只在 ROI 里计分：mask 0 = 不在任何 ROI_CHECK 里，预测和 GT 都清零。

在 BiomedParse 仓库根目录、biomedparse 环境里运行：
  python test_batch_cell.py --selftest                          # 只测打分逻辑，几秒，不用 GPU
  python test_batch_cell.py --check                             # 只检查数据，不加载模型
  CUDA_VISIBLE_DEVICES=1 python test_batch_cell.py --limit 20   # 随机抽 20 张试跑
  CUDA_VISIBLE_DEVICES=1 python test_batch_cell.py              # 全量（重跑同一个 --out 会跳过已预测的 tile）

prompt：默认 = 我们的类名（和我们的模型问同一个问题）；--trained-prompts = 最接近的 v1 训练目标
（PanNuke：neoplastic / epithelial / inflammatory / connective tissue cells）。注意 v1 只有一个 "inflammatory cells"，
所以 --trained-prompts 时 5 个免疫细胞类用同一张概率图，类别投票会把这些细胞全给排在前面的那一类 ——
它们各自的 PQ 没有意义，看 binary 指标和有自己 prompt 的类（epithelial、stromal）。

数据：<data>/image/<tile>.jpg + mask/<tile>.png + inst/<tile>.tif
（VPR_LISA_BASED/scripts/ukk_col/cell_extract_with_args.groovy 的输出）：
  mask 0 = ROI 外（不计分），10 = ROI 里没有细胞，1..9 = 细胞类（1 neutrophils 2 tumor 3 lymphocytes 4 plasma
  5 eosinophils 6 connective 7 mitotic 8 benign 9 macrophage）；inst = 每个细胞一个编号

输出（--out）：config.json、pred/<tile>.npz（8 个二值 mask + 类别图）、records.jsonl、summary.json、vis/
"""
import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image

DEFAULT_DATA = "/data/02_ZW/New_train/testds/Test_DS/Cell/ZW/07_Tiles"
DEFAULT_VPR = "/data/02_ZW/New_train/VPR_LISA_BASED"
UNANNOTATED, ROI_EMPTY = 0, 10

# 通道: (我们的类名 = VPR targets.CELL_DISPLAY, mask 标签值, 最接近的 v1 训练目标；None = v1 训练数据里没有)
# 通道顺序 = VPR model.cell_branch.class_channels()（vocab.seg_cell_dict 按 id 排；benign 8 并进 epithelial）
CLASSES = {
    1: ("neutrophils",      (1,),   ("inflammatory cells",)),
    2: ("epithelial cells", (2, 8), ("neoplastic cells", "epithelial cells")),   # tumor + benign
    3: ("lymphocytes",      (3,),   ("inflammatory cells",)),
    4: ("plasma cells",     (4,),   ("inflammatory cells",)),
    5: ("eosinophils",      (5,),   ("inflammatory cells",)),
    6: ("stromal cells",    (6,),   ("connective tissue cells",)),
    7: ("mitotic figures",  (7,),   None),
    8: ("macrophages",      (9,),   ("inflammatory cells",)),
}
CHANNELS = sorted(CLASSES)
LUT = np.zeros(256, np.int64)                     # mask 标签 -> 通道
for _c, (_n, _labels, _p) in CLASSES.items():
    for _v in _labels:
        LUT[_v] = _c
PAL = np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255],
                [0, 160, 0], [255, 128, 0], [128, 0, 255]], np.uint8)


def load_instance_metrics(vpr_repo):
    """VPR 的 utils/instance_metrics.py，按文件加载（不导入 VPR 的其它模块）。"""
    path = os.path.join(vpr_repo, "utils", "instance_metrics.py")
    if not os.path.isfile(path):
        sys.exit(f"--vpr-repo: 找不到 {path}")
    spec = importlib.util.spec_from_file_location("vpr_instance_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- 数据 ----------------

def list_tiles(data):
    d = {k: os.path.join(data, k) for k in ("image", "mask", "inst")}
    for k, v in d.items():
        if not os.path.isdir(v):
            sys.exit(f"文件夹不存在：{v}")
    stems = {k: {os.path.splitext(f)[0]: os.path.join(d[k], f) for f in os.listdir(d[k])} for k in ("mask", "inst")}
    out, missing = [], 0
    for f in sorted(os.listdir(d["image"])):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        if stem in stems["mask"] and stem in stems["inst"]:
            out.append((stem, os.path.join(d["image"], f), stems["mask"][stem], stems["inst"][stem]))
        else:
            missing += 1
    return out, missing


def read_gt(mask_path, inst_path):
    """(valid, GT 通道图, GT 实例图)，ROI 外清零 —— 与 VPR test_batch_cells.read_gt 相同。"""
    sem = np.array(Image.open(mask_path)).astype(np.int64)
    inst = np.array(Image.open(inst_path)).astype(np.int64)
    if sem.shape != inst.shape:
        raise ValueError(f"{mask_path}: mask {sem.shape} vs inst {inst.shape}")
    unknown = set(np.unique(sem).tolist()) - set(range(0, 11))
    if unknown:
        raise ValueError(f"{mask_path}: 未定义的标签 {sorted(unknown)}")
    valid = sem != UNANNOTATED
    raw = np.where((sem >= 1) & (sem <= 9), sem, 0)
    return valid, LUT[raw] * valid, inst * valid


def check(tiles, missing, n=50):
    print(f"配对成功 {len(tiles)} 张；缺 mask 或 inst 的图 {missing} 张")
    if not tiles:
        return
    sample = random.Random(0).sample(tiles, min(n, len(tiles)))
    sizes, pix = Counter(), Counter()
    for _s, ip, mp, sp in sample:
        im = Image.open(ip)
        valid, ch, inst = read_gt(mp, sp)
        sizes[f"image {im.size[0]}x{im.size[1]} {im.mode} / mask {valid.shape[1]}x{valid.shape[0]}"] += 1
        pix["ROI 外"] += int((~valid).sum())
        for c in CHANNELS:
            pix[CLASSES[c][0]] += int((ch == c).sum())
        pix["细胞数"] += len(np.unique(inst)) - 1
    print(f"随机抽 {len(sample)} 张：")
    for s, k in sizes.most_common():
        print(f"  {s}: {k}")
    for k, v in pix.items():
        print(f"  {k:<18}{v:>12}")


# ---------------- 预测 ----------------

def prompts_for(template, trained):
    """通道 -> 这一类的 prompt 列表（多个 prompt 的概率取最大值）。"""
    out = {}
    for c in CHANNELS:
        name, _labels, tp = CLASSES[c]
        out[c] = [template.format(name=p) for p in (tp if (trained and tp) else (name,))]
    return out


def predict_tile(model, image, prompts, tile, joint, thr):
    """(8 个二值 mask (8, H, W) bool, 类别通道图 (H, W) uint8)。相同的 prompt 只推理一次。"""
    import torch
    from test_batch import predict              # BiomedParse 组织评测的同一个推理函数
    unique = sorted({p for ps in prompts.values() for p in ps})
    probs = dict(zip(unique, predict(model, image, unique, tile, joint)))
    stack = np.stack([np.max([probs[p] for p in prompts[c]], axis=0) for c in CHANNELS])     # (8, H, W)
    masks = stack > thr
    logits = torch.from_numpy(stack - thr)
    best, idx = logits.max(0)
    typed = torch.where(best > 0, torch.as_tensor(CHANNELS)[idx], torch.zeros_like(idx))
    return masks, typed.numpy().astype(np.uint8)


# ---------------- 打分（与 VPR test_batch_cells.score 相同） ----------------

def score(tiles, pred_dir, im, min_cell_px):
    hover = im.HoverMeter()
    rows = []
    for stem, _ip, mp, sp in tiles:
        f = os.path.join(pred_dir, stem + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        valid, gt_ch, gt_inst = read_gt(mp, sp)
        masks = np.unpackbits(z["masks"], count=len(CHANNELS) * valid.size).reshape((len(CHANNELS),) + valid.shape)
        masks = masks.astype(bool)
        typed = z["typed"].astype(np.int64) * valid
        hover.update_tile("model", typed, gt_inst, gt_ch, margins=(0, 16))
        for k, c in enumerate(CHANNELS):
            gt = gt_ch == c
            p = masks[k] & valid
            inter, ps, gs = int((p & gt).sum()), int(p.sum()), int(gt.sum())
            rows.append({"image": stem, "slide": stem.split(" [")[0], "class": CLASSES[c][0],
                         "present": gs >= min_cell_px, "drew": ps > 0, "inter": inter, "pred_px": ps, "gt_px": gs,
                         "dice": 2.0 * inter / (ps + gs) if ps + gs else 1.0})
    return hover, rows


def summarize_pixels(rows):
    per = {}
    for c in CHANNELS:
        name = CLASSES[c][0]
        pres = [r for r in rows if r["class"] == name and r["present"]]
        absn = [r for r in rows if r["class"] == name and not r["present"]]
        inter, ps, gs = (sum(r[k] for r in pres) for k in ("inter", "pred_px", "gt_px"))
        per[name] = {"n_present": len(pres), "empty_present": sum(not r["drew"] for r in pres),
                     "mean_dice": float(np.mean([r["dice"] for r in pres])) if pres else None,
                     "pooled_dice": 2.0 * inter / (ps + gs) if ps + gs else None,
                     "precision": inter / ps if ps else None, "recall": inter / gs if gs else None,
                     "n_absent": len(absn),
                     "drew_on_absent": float(np.mean([r["drew"] for r in absn])) if absn else None}
    md = [v["mean_dice"] for v in per.values() if v["mean_dice"] is not None]
    pdc = [v["pooled_dice"] for v in per.values() if v["pooled_dice"] is not None]
    return {"per_class": per, "macro_mean_dice": float(np.mean(md)) if md else None,
            "macro_pooled_dice": float(np.mean(pdc)) if pdc else None,
            "n_tiles": len({r["image"] for r in rows}), "n_slides": len({r["slide"] for r in rows})}


def print_pixels(s, min_cell_px, prompts):
    f = lambda x: "    -" if x is None else f"{x:.3f}"                                     # noqa: E731
    print(f"\nPIXEL Dice per class (tiles with >= {min_cell_px} GT px of the class in the ROI; 'absent' = the others)")
    print(f"{'class':<18}{'n':>6}{'empty':>7}{'mean':>8}{'pooled':>8}{'P':>7}{'R':>7}{'absent':>8}{'drew%':>7}  prompt")
    for c in CHANNELS:
        name = CLASSES[c][0]
        v = s["per_class"][name]
        drew = "    -" if v["drew_on_absent"] is None else f"{100 * v['drew_on_absent']:5.1f}"
        print(f"{name:<18}{v['n_present']:>6}{v['empty_present']:>7}{f(v['mean_dice']):>8}{f(v['pooled_dice']):>8}"
              f"{f(v['precision']):>7}{f(v['recall']):>7}{v['n_absent']:>8}{drew:>7}  {' | '.join(prompts[c])}")
    print(f"{'macro':<31}{f(s['macro_mean_dice']):>8}{f(s['macro_pooled_dice']):>8}")
    print(f"{s['n_tiles']} tiles from {s['n_slides']} slides")


def save_vis(path, image, gt_ch, typed, valid, size=512):
    import cv2

    def paint(ch):
        out = image.copy()
        m = ch > 0
        out[m] = (out[m] * 0.4 + PAL[ch[m] % len(PAL)] * 0.6).astype(np.uint8)
        out[~valid] = (out[~valid] * 0.4 + 128 * 0.6).astype(np.uint8)
        return out
    row = np.concatenate([cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)
                          for p in (image, paint(gt_ch), paint(typed * valid))], axis=1)
    Image.fromarray(row).save(path)


# ---------------- 自测 ----------------

def selftest(im):
    import tempfile
    rng = np.random.default_rng(0)
    H = 256
    sem = np.full((H, H), ROI_EMPTY, np.uint8)
    sem[:, :32] = UNANNOTATED                       # ROI 外
    inst = np.zeros((H, H), np.uint16)
    yy, xx = np.mgrid[:H, :H]
    k = 0
    for cy in range(24, H, 40):
        for cx in range(56, H, 40):
            k += 1
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= 9 ** 2
            inst[disc] = k
            sem[disc] = rng.choice([1, 2, 3, 6, 8, 9])
    with tempfile.TemporaryDirectory() as d:
        for sub in ("image", "mask", "inst"):
            os.makedirs(os.path.join(d, sub))
        Image.fromarray(np.zeros((H, H, 3), np.uint8)).save(os.path.join(d, "image", "t.jpg"))
        Image.fromarray(sem, "L").save(os.path.join(d, "mask", "t.png"))
        Image.fromarray(inst).save(os.path.join(d, "inst", "t.tif"))
        tiles, _ = list_tiles(d)
        valid, gt_ch, _ = read_gt(tiles[0][2], tiles[0][3])
        assert int((~valid).sum()) == H * 32 and set(np.unique(gt_ch)) <= {0, 1, 2, 3, 6, 8}
        assert (gt_ch[sem == 8] == 2).all() and (gt_ch[sem == 9] == 8).all()      # benign -> epithelial, macro -> 8
        pdir = os.path.join(d, "pred")
        os.makedirs(pdir)
        for tag, masks, typed in (("perfect", np.stack([gt_ch == c for c in CHANNELS]), gt_ch),
                                  ("empty", np.zeros((8, H, H), bool), np.zeros_like(gt_ch))):
            np.savez_compressed(os.path.join(pdir, "t.npz"), masks=np.packbits(masks.ravel()),
                                typed=typed.astype(np.uint8))
            hover, rows = score(tiles, pdir, im, min_cell_px=50)
            s, h = summarize_pixels(rows), hover.summary()
            if tag == "perfect":
                assert abs(s["macro_mean_dice"] - 1) < 1e-9, s
                assert h["binary"]["model"]["pq"] > 0.99 and h["typed"]["model"]["mpq"] > 0.99, h["typed"]["model"]
            else:
                assert s["macro_mean_dice"] == 0 and h["binary"]["model"]["pq"] == 0
    # 相同 prompt 合并、类别投票
    pr = prompts_for("{name}", True)
    assert pr[1] == pr[3] == ["inflammatory cells"] and pr[2] == ["neoplastic cells", "epithelial cells"]
    assert pr[7] == ["mitotic figures"] and prompts_for("{name} in pathology", False)[6] == ["stromal cells in pathology"]
    print("selftest OK")


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="BiomedParse v1 细胞分割批量评测")
    ap.add_argument("--data", default=DEFAULT_DATA, help="包含 image/ mask/ inst/ 的目录")
    ap.add_argument("--out", default="out/wns_cells", help="结果输出目录（不同设置请用不同目录）")
    ap.add_argument("--vpr-repo", default=DEFAULT_VPR, help="VPR_LISA_BASED 仓库（用它的 utils/instance_metrics.py）")
    ap.add_argument("--tile", type=int, default=0, help="切块边长；0 = 整张图直接推理")
    ap.add_argument("--template", default="{name}", help='prompt 模板，例如 "{name} in pathology"')
    ap.add_argument("--trained-prompts", action="store_true", help="改用最接近的 v1 训练目标作 prompt")
    ap.add_argument("--joint", action="store_true", help="一张图的 prompt 一次推理（更快，但 prompt 之间会相互影响）")
    ap.add_argument("--thr", type=float, default=0.5, help="概率阈值")
    ap.add_argument("--min-cell-px", type=int, default=200, help="这类 GT 像素 >= 此值的 tile 才算 present")
    ap.add_argument("--limit", type=int, default=0, help="随机抽 N 张（固定种子），0 = 全部")
    ap.add_argument("--save-vis", type=int, default=0, help="保存前 N 张的对比图（原图 | GT | 预测）")
    ap.add_argument("--check", action="store_true", help="只检查数据，不加载模型")
    ap.add_argument("--selftest", action="store_true", help="只测打分逻辑")
    a = ap.parse_args()

    im = load_instance_metrics(a.vpr_repo)
    if a.selftest:
        selftest(im)
        return
    tiles, missing = list_tiles(a.data)
    if a.check:
        check(tiles, missing)
        return
    if not tiles:
        sys.exit("没有配对成功的 tile，先跑 --check")
    if "{name}" not in a.template:
        sys.exit("--template 里必须包含 {name}")
    if not os.path.exists("configs/biomedparse_inference.yaml"):
        sys.exit("请在 BiomedParse 仓库根目录运行")
    if a.limit:
        tiles = sorted(random.Random(0).sample(tiles, min(a.limit, len(tiles))))
    prompts = prompts_for(a.template, a.trained_prompts)

    pred_dir = os.path.join(a.out, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    cfg_path = os.path.join(a.out, "config.json")
    setting = {"thr": a.thr, "tile": a.tile, "joint": a.joint, "prompts": {CLASSES[c][0]: prompts[c] for c in CHANNELS}}
    if os.path.exists(cfg_path):                 # 续跑：已存的预测必须是同一个设置
        old = json.load(open(cfg_path))
        if {k: old.get(k) for k in setting} != setting:
            sys.exit(f"{a.out} 里已有另一组设置的预测（{ {k: old.get(k) for k in setting} }），请换 --out")
    with open(cfg_path, "w") as f:
        json.dump({**vars(a), **setting, "n_tiles": len(tiles)}, f, ensure_ascii=False, indent=2)
    todo = [t for t in tiles if not os.path.exists(os.path.join(pred_dir, t[0] + ".npz"))]
    print(f"{len(tiles)} 张 tile（{missing} 张缺 mask/inst）；已预测 {len(tiles) - len(todo)}；阈值 {a.thr}；"
          f"prompt: {json.dumps(setting['prompts'], ensure_ascii=False)}")

    if todo:
        from test_batch import load_model     # BiomedParse 组织评测的同一个加载函数
        from tqdm import tqdm
        model = load_model()
        if a.save_vis:
            os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
        for i, (stem, ip, mp, sp) in enumerate(tqdm(todo, dynamic_ncols=True)):
            image = np.array(Image.open(ip).convert("RGB"))
            masks, typed = predict_tile(model, image, prompts, a.tile, a.joint, a.thr)
            np.savez_compressed(os.path.join(pred_dir, stem + ".npz"), masks=np.packbits(masks.ravel()), typed=typed)
            if i < a.save_vis:
                valid, gt_ch, _ = read_gt(mp, sp)
                save_vis(os.path.join(a.out, "vis", f"{i:05d}.jpg"), image, gt_ch, typed, valid)

    hover, rows = score(tiles, pred_dir, im, a.min_cell_px)
    with open(os.path.join(a.out, "records.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pix, hs = summarize_pixels(rows), hover.summary()
    names = {c: CLASSES[c][0] for c in CHANNELS}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"pixel": pix, "hover": hs, "settings": setting}, f, ensure_ascii=False, indent=2, default=str)
    print_pixels(pix, a.min_cell_px, prompts)
    print("\nNUCLEUS instance metrics (VPR utils/instance_metrics.HoverMeter, inside the ROI; @in16 = without the outer 16 px)")
    for line in im.summary_lines("cells", hs, names):
        print(line)
    print(f"\n结果在 {a.out}/")


if __name__ == "__main__":
    main()
