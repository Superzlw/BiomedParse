#!/usr/bin/env python
"""BiomedParse v1 组织分割批量评测（UKK_Col 数据集）。

在 BiomedParse 仓库根目录、biomedparse 环境里运行：
  python test_batch.py --selftest                          # 只测 Dice 计算逻辑，几秒，不用 GPU
  python test_batch.py --check                             # 只检查数据（配对/尺寸/标签分布），不加载模型
  CUDA_VISIBLE_DEVICES=1 python test_batch.py --limit 50   # 随机抽 50 张试跑
  CUDA_VISIBLE_DEVICES=1 python test_batch.py              # 全量

评测协议（与 BiomedParse 论文一致）：每张图只对 GT 中出现的类别发 prompt，Dice 只在该类别出现的图上统计，
衡量的是"已知目标存在时分得准不准"，不衡量"能否判断目标不存在"。
加 --all-prompts 则每张图都发全部 8 个 prompt，额外报告把误检也算进去的 Dice（all 列）。

像素规则：未标注(0) 永远不参与计算；背景(11) 不作为类别评测，但模型在背景上的预测默认算误报；
加 --ignore-background 则背景像素也完全不参与计算。

输出（--out 目录）：config.json（参数与 prompt）、per_tile.csv（每张图每个类别的像素数与 Dice）、
summary.csv（每类与各组的汇总）、vis/（--save-vis 时的对比图：原图 | GT | 预测）。
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image

DEFAULT_DATA = "/data/02_ZW/Raw_Dataset/VPR/Test_DS/UKK_Col/07_Tiles"

# 类别名: (mask 标签值, 最接近的 v1 训练目标；None = v1 训练数据里没有任何相关概念)
# v1 病理训练目标只有细胞核级（PanNuke 的 4 类细胞）和结肠腺体（GlaS），没有组织区域级的类别，
# 所以 related 组只是概念相近，并不等于训练过。可以按需修改这里的对应关系。
CLASSES = {
    "epithelial tissue":        ((1,),      "epithelial cells"),
    "colorectal mucosa":        ((2,),      "glandular structure"),
    "connective tissue":        ((3, 4, 6), "connective tissue cells"),
    "muscle tissue":            ((5,),      None),
    "dense lymphoid aggregate": ((7,),      "inflammatory cells"),
    "necrotic debris":          ((8,),      None),
    "erythrocytes":             ((9,),      None),
    "mucin":                    ((10,),     None),
}
UNANNOTATED, BACK = 0, 11

NAMES = list(CLASSES)
RELATED = [c for c in NAMES if CLASSES[c][1] is not None]
ABSENT = [c for c in NAMES if CLASSES[c][1] is None]
GROUPS = {"related-in-training": RELATED, "absent-from-training": ABSENT, "ALL": NAMES}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLORS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                   [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.uint8)
NAN = float("nan")


# ---------------- Dice 计算（--selftest 测的就是这几个函数，主流程也调用同样的函数） ----------------

def dice(tp, p, g):
    return 2.0 * tp / (p + g) if (p + g) > 0 else NAN


def valid_region(mask, ignore_bg):
    """参与计算的像素：去掉未标注；--ignore-background 时再去掉背景。"""
    v = mask != UNANNOTATED
    if ignore_bg:
        v &= mask != BACK
    return v


def tile_counts(mask, pred, labels, valid):
    """返回 (tp, 预测像素数, GT 像素数)，只在 valid 区域内计数。"""
    gt = np.isin(mask, labels) & valid
    pr = pred & valid
    return int((pr & gt).sum()), int(pr.sum()), int(gt.sum())


def new_stat():
    return {"n": 0, "dsum": 0.0, "tp": 0, "p": 0, "g": 0, "tp_all": 0, "p_all": 0, "g_all": 0}


def add(stat, tp, p, g):
    stat["tp_all"] += tp
    stat["p_all"] += p
    stat["g_all"] += g
    if g > 0:                       # 只有 GT 里有这个类别的图才进入 mean / pooled
        stat["n"] += 1
        stat["dsum"] += dice(tp, p, g)
        stat["tp"] += tp
        stat["p"] += p
        stat["g"] += g


def summarize(stats, all_prompts):
    per_class = {}
    for c in NAMES:
        s = stats[c]
        per_class[c] = {
            "tiles": s["n"],
            "mean": s["dsum"] / s["n"] if s["n"] else NAN,
            "pooled": dice(s["tp"], s["p"], s["g"]),
            "all": dice(s["tp_all"], s["p_all"], s["g_all"]) if all_prompts else NAN,
        }
    groups = {}
    for g, members in GROUPS.items():
        cs = [c for c in members if stats[c]["n"] > 0]      # 测试集里没出现过的类别不进入总体

        def tot(k):
            return sum(stats[c][k] for c in cs)

        groups[g] = {
            "classes": len(cs),
            "macro": float(np.mean([per_class[c]["mean"] for c in cs])) if cs else NAN,
            "micro": dice(tot("tp"), tot("p"), tot("g")),
            "micro_all": dice(tot("tp_all"), tot("p_all"), tot("g_all")) if all_prompts else NAN,
        }
    return per_class, groups


def selftest():
    m = np.array([[1, 1, 2, 0],
                  [3, 4, 6, 11],
                  [5, 5, 11, 11]])
    # epithelial：GT 2 个像素；预测命中 1 个、漏 1 个，另在未标注(0)和背景(11)上各误报 1 个
    pred = np.zeros(m.shape, bool)
    pred[0, 0] = pred[0, 3] = pred[1, 3] = True
    assert tile_counts(m, pred, (1,), valid_region(m, False)) == (1, 2, 2)   # 未标注不计，背景算误报
    assert tile_counts(m, pred, (1,), valid_region(m, True)) == (1, 1, 2)    # 背景也不计
    # connective = 标签 3/4/6 合并；整张图都预测成它
    allp = np.ones(m.shape, bool)
    assert tile_counts(m, allp, (3, 4, 6), valid_region(m, False)) == (3, 11, 3)
    assert tile_counts(m, allp, (3, 4, 6), valid_region(m, True)) == (3, 8, 3)

    st = {c: new_stat() for c in NAMES}
    add(st["epithelial tissue"], 4, 5, 5)    # 单张 Dice 0.8
    add(st["epithelial tissue"], 4, 5, 5)    # 0.8
    add(st["epithelial tissue"], 0, 3, 0)    # GT 里没有：不进 mean/pooled，只进 all（误检）
    add(st["mucin"], 1, 5, 5)                # 0.2
    pc, gr = summarize(st, all_prompts=True)

    def close(a, b):
        return abs(a - b) < 1e-9

    e = pc["epithelial tissue"]
    assert e["tiles"] == 2 and close(e["mean"], 0.8) and close(e["pooled"], 0.8)
    assert close(e["all"], 16 / 23)                          # 2*8 / (13+10)
    assert close(gr["ALL"]["macro"], 0.5)                    # (0.8 + 0.2) / 2
    assert close(gr["ALL"]["micro"], 18 / 30)                # 2*(8+1) / ((10+5)+(10+5))
    assert gr["related-in-training"]["classes"] == 1 and gr["absent-from-training"]["classes"] == 1
    assert pc["muscle tissue"]["tiles"] == 0 and pc["muscle tissue"]["mean"] != pc["muscle tissue"]["mean"]
    print("selftest OK")


# ---------------- 数据 ----------------

def load_mask(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:   # 用 RGB(A) 存的标签图：前三个通道必须相同
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError(f"mask 是彩色图，无法直接读成标签: {path}")
        a = a[..., 0]
    return a.astype(np.int64)


def list_by_stem(d):
    out = {}
    for f in os.listdir(d):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT:
            out[stem] = os.path.join(d, f)
    return out


def pair_files(img_dir, mask_dir):
    imgs, masks = list_by_stem(img_dir), list_by_stem(mask_dir)
    common = sorted(imgs.keys() & masks.keys())
    return [(imgs[k], masks[k]) for k in common], imgs.keys() - masks.keys(), masks.keys() - imgs.keys()


def label_names():
    names = {UNANNOTATED: "unannotated", BACK: "background"}
    for c, (labels, _) in CLASSES.items():
        for v in labels:
            names[v] = c
    return names


def check(pairs, only_img, only_mask, n=50):
    print(f"配对成功 {len(pairs)} 对；只有图像 {len(only_img)}；只有 mask {len(only_mask)}")
    for title, s in (("只有图像", only_img), ("只有 mask", only_mask)):
        if s:
            print(f"  {title} 示例: {sorted(s)[:3]}")
    if not pairs:
        return
    sample = random.Random(0).sample(pairs, min(n, len(pairs)))
    sizes, pix, tiles = Counter(), Counter(), Counter()
    for ip, mp in sample:
        im, m = Image.open(ip), load_mask(mp)
        sizes[f"image {im.size[0]}x{im.size[1]} {im.mode} / mask {m.shape[1]}x{m.shape[0]}"] += 1
        u, cnt = np.unique(m, return_counts=True)
        for v, k in zip(u.tolist(), cnt.tolist()):
            pix[v] += k
            tiles[v] += 1
    print(f"随机抽 {len(sample)} 张的尺寸:")
    for s, k in sizes.most_common():
        print(f"  {s}: {k}")
    names, total = label_names(), sum(pix.values())
    print("标签分布（像素占比 / 出现在几张图里）:")
    for v in sorted(pix):
        print(f"  {v:>3} {names.get(v, '??? 未定义'):<26}{pix[v] / total:8.2%}  {tiles[v]}/{len(sample)}")
    unknown = sorted(v for v in pix if v not in names)
    if unknown:
        print(f"⚠ 发现未定义的标签值 {unknown}，先确认 mask 格式再跑评测")


# ---------------- 模型 ----------------

def load_model():
    import torch
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES

    opt = init_distributed(load_opt_from_config_files(["configs/biomedparse_inference.yaml"]))
    model = BaseModel(opt, build_model(opt)).from_pretrained("hf_hub:microsoft/BiomedParse").eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    return model


def predict(model, image, prompts, tile, joint):
    """返回 (len(prompts), H, W) 概率图。tile>0 时按 tile×tile 切块推理再拼回。"""
    from inference_utils.inference import interactive_infer_image
    H, W = image.shape[:2]
    step = tile if tile > 0 else max(H, W)
    out = np.zeros((len(prompts), H, W), np.float32)
    for y in range(0, H, step):
        for x in range(0, W, step):
            crop = Image.fromarray(image[y:y + step, x:x + step])
            if joint:
                out[:, y:y + step, x:x + step] = interactive_infer_image(model, crop, prompts)
            else:   # 默认每个 prompt 单独推理，结果互不影响
                for i, p in enumerate(prompts):
                    out[i, y:y + step, x:x + step] = interactive_infer_image(model, crop, [p])[0]
    return out


def build_prompts(template, trained):
    out = {}
    for c in NAMES:
        name = CLASSES[c][1] if (trained and CLASSES[c][1]) else c
        out[c] = template.format(name=name)
    return out


def save_vis(path, image, mask, valid, probs, prompted, thr):
    gt = np.full(image.shape, 128, np.uint8)      # 灰 = 不参与计算
    gt[valid] = 0                                  # 黑 = 参与计算的背景
    for i, c in enumerate(NAMES):
        gt[np.isin(mask, CLASSES[c][0])] = COLORS[i]
    pr = np.zeros_like(image)
    if prompted:
        hit, best = probs.max(0) > thr, probs.argmax(0)
        for j, c in enumerate(prompted):
            pr[hit & (best == j)] = COLORS[NAMES.index(c)]
    Image.fromarray(np.concatenate([image, gt, pr], axis=1)).save(path)


# ---------------- 报告 ----------------

def fmt(x):
    return f"{x:8.4f}" if x == x else f"{'-':>8}"


def num(x):
    return f"{x:.4f}" if x == x else ""


def report(per_class, groups, prompts, all_prompts, out_dir):
    pw = max(len(p) for p in prompts.values()) + 2
    cols = ["mean", "pooled"] + (["all"] if all_prompts else [])
    head = f"{'class':<26}{'group':<9}{'prompt':<{pw}}{'tiles':>6}" + "".join(f"{c:>8}" for c in cols)
    print("\n" + head + "\n" + "-" * len(head))
    rows = []
    for c in NAMES:
        r = per_class[c]
        grp = "related" if c in RELATED else "absent"
        vals = [r["mean"], r["pooled"]] + ([r["all"]] if all_prompts else [])
        print(f"{c:<26}{grp:<9}{prompts[c]:<{pw}}{r['tiles']:>6}" + "".join(fmt(v) for v in vals))
        rows.append(["class", c, grp, prompts[c], r["tiles"], num(r["mean"]), num(r["pooled"]), num(r["all"])])
    print("-" * len(head))
    for g, r in groups.items():
        vals = [r["macro"], r["micro"]] + ([r["micro_all"]] if all_prompts else [])
        label = f"({r['classes']} classes)"
        print(f"{g:<26}{'':<9}{label:<{pw}}{'':>6}" + "".join(fmt(v) for v in vals))
        rows.append(["group", g, "", "", r["classes"], num(r["macro"]), num(r["micro"]), num(r["micro_all"])])
    print("\nmean   = 每张图 Dice 的平均；组那几行是各类 mean 再取平均（macro，每类权重相同）")
    print("pooled = 所有图的像素合并后算一个 Dice；组那几行是跨类别合并（micro，大类权重大）")
    if all_prompts:
        print("all    = 同 pooled，但把 GT 里没有该类别的图也算进去（误检会拉低它）")
    print("tiles  = GT 里出现该类别的图数；没出现过的类别不进入总体")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "name", "group", "prompt", "tiles_or_classes",
                    "mean_or_macro", "pooled_or_micro", "all_pooled"])
        w.writerows(rows)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="BiomedParse v1 组织分割批量评测")
    ap.add_argument("--data", default=DEFAULT_DATA, help="包含 image/ 和 mask/ 的目录")
    ap.add_argument("--out", default="out/ukk_col_tissue", help="结果输出目录（不同设置请用不同目录）")
    ap.add_argument("--tile", type=int, default=0, help="切块边长；0 = 整张图直接推理")
    ap.add_argument("--template", default="{name}", help='prompt 模板，例如 "{name} in colon pathology"')
    ap.add_argument("--trained-prompts", action="store_true", help="related 组改用最接近的 v1 训练目标作 prompt")
    ap.add_argument("--all-prompts", action="store_true", help="每张图都发全部 8 个 prompt，额外报告含误检的 Dice")
    ap.add_argument("--joint", action="store_true", help="一张图的 prompt 一次推理（更快，但 prompt 之间会相互影响）")
    ap.add_argument("--ignore-background", action="store_true", help="背景(11)像素也不参与计算")
    ap.add_argument("--thr", type=float, default=0.5, help="概率阈值")
    ap.add_argument("--limit", type=int, default=0, help="随机抽 N 张（固定种子），0 = 全部")
    ap.add_argument("--save-vis", type=int, default=0, help="保存前 N 张的对比图（原图 | GT | 预测）")
    ap.add_argument("--check", action="store_true", help="只检查数据，不加载模型")
    ap.add_argument("--selftest", action="store_true", help="只测 Dice 计算逻辑")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    pairs, only_img, only_mask = pair_files(os.path.join(a.data, "image"), os.path.join(a.data, "mask"))
    if a.check:
        check(pairs, only_img, only_mask)
        return
    if not pairs:
        sys.exit("没有配对成功的图像 / mask，先跑 --check")
    if "{name}" not in a.template:
        sys.exit("--template 里必须包含 {name}")
    if not os.path.exists("configs/biomedparse_inference.yaml"):
        sys.exit("请在 BiomedParse 仓库根目录运行")
    if a.limit:
        pairs = sorted(random.Random(0).sample(pairs, min(a.limit, len(pairs))))

    prompts = build_prompts(a.template, a.trained_prompts)
    os.makedirs(a.out, exist_ok=True)
    if a.save_vis:
        os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({**vars(a), "prompts": prompts, "n_pairs": len(pairs),
                   "unpaired_images": len(only_img), "unpaired_masks": len(only_mask)},
                  f, ensure_ascii=False, indent=2)
    print(f"{len(pairs)} 张图；prompt: {json.dumps(prompts, ensure_ascii=False)}")

    model = load_model()
    from tqdm import tqdm

    stats = {c: new_stat() for c in NAMES}
    n_eval, skipped, n_vis = 0, [], 0
    with open(os.path.join(a.out, "per_tile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "prompt", "present", "gt_px", "pred_px", "tp", "dice"])
        for ip, mp in tqdm(pairs, dynamic_ncols=True):
            try:
                image = np.array(Image.open(ip).convert("RGB"))
                mask = load_mask(mp)
            except (OSError, ValueError) as e:
                skipped.append((ip, f"读取失败: {e}"))
                continue
            if image.shape[:2] != mask.shape:
                skipped.append((ip, f"尺寸不一致 image {image.shape[:2]} / mask {mask.shape}"))
                continue
            valid = valid_region(mask, a.ignore_background)
            present = [c for c in NAMES if np.isin(mask, CLASSES[c][0]).any()]
            todo = NAMES if a.all_prompts else present
            if not todo:
                skipped.append((ip, "没有任何评测类别（只有未标注/背景）"))
                continue
            probs = predict(model, image, [prompts[c] for c in todo], a.tile, a.joint)
            for i, c in enumerate(todo):
                tp, p, g = tile_counts(mask, probs[i] > a.thr, CLASSES[c][0], valid)
                add(stats[c], tp, p, g)
                w.writerow([os.path.basename(ip), c, prompts[c], int(g > 0), g, p, tp,
                            f"{dice(tp, p, g):.4f}" if g > 0 else ""])
            n_eval += 1
            if n_vis < a.save_vis:
                stem = os.path.splitext(os.path.basename(ip))[0]
                save_vis(os.path.join(a.out, "vis", f"{stem}.png"), image, mask, valid, probs, todo, a.thr)
                n_vis += 1

    per_class, groups = summarize(stats, a.all_prompts)
    report(per_class, groups, prompts, a.all_prompts, a.out)
    print(f"\n评测 {n_eval} 张；跳过 {len(skipped)} 张；结果在 {a.out}/")
    for ip, why in skipped[:5]:
        print(f"  跳过 {os.path.basename(ip)}: {why}")


if __name__ == "__main__":
    main()
