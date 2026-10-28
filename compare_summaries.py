#!/usr/bin/env python
"""并排比较多个 test_batch.py 输出目录里的 summary.csv。

在 BiomedParse 仓库根目录运行：
  python compare_summaries.py out/ukk_col_names out/charite_col_names out/kameda_col_names out/lmu_col_names out/wns_col_names
  python compare_summaries.py out/*_col_names --thr 0.2 --metric mean --save out/compare_thr0.2_mean.csv

每类给出所选的 Dice（pooled 或 mean），各组同时给出 macro 和 micro；
下面另附每类的 tiles 数（GT 里含该类别的图数），n 很小的格子不要太当真。
某个目录没有所选阈值时，那一列留空（例如 UKK 只评了 0.2–0.5，没有 0.1）。
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="并排比较多个 test_batch.py 的结果")
    ap.add_argument("dirs", nargs="+", help="test_batch.py 的 --out 目录")
    ap.add_argument("--thr", type=float, default=0.5, help="比较哪个阈值")
    ap.add_argument("--metric", choices=["pooled", "mean"], default="pooled", help="每类用哪个 Dice")
    ap.add_argument("--save", default="", help="把 Dice 表另存为 CSV（可选）")
    a = ap.parse_args()

    col = "pooled_or_micro" if a.metric == "pooled" else "mean_or_macro"
    dice, tiles, order = {}, {}, []
    for d in a.dirs:
        name = os.path.basename(os.path.normpath(d))
        s = pd.read_csv(os.path.join(d, "summary.csv"))
        s = s[(s["thr"] - a.thr).abs() < 1e-9]
        if s.empty:
            print(f"⚠ {d} 里没有 thr={a.thr} 的结果，这一列留空")
            dice[name], tiles[name] = pd.Series(dtype=float), pd.Series(dtype=float)
            continue
        cls = s[s["level"] == "class"].set_index("name")
        grp = s[s["level"] == "group"].set_index("name")
        dice[name] = pd.concat([cls[col],
                                grp["mean_or_macro"].rename(lambda g: f"{g} [macro]"),
                                grp["pooled_or_micro"].rename(lambda g: f"{g} [micro]")])
        tiles[name] = cls["tiles_or_classes"]
        order += [r for r in dice[name].index if r not in order]

    dice = pd.DataFrame(dice).reindex(order)
    tiles = pd.DataFrame(tiles).reindex([r for r in order if "[" not in r])
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(f"\nDice（每类 = {a.metric}，thr = {a.thr}）")
        print(dice.to_string(na_rep="-", float_format="%.3f"))
        print("\ntiles（GT 里含该类别的图数）")
        print(tiles.to_string(na_rep="-", float_format="%.0f"))
    if a.save:
        dice.to_csv(a.save)
        print(f"\n已保存 {a.save}")


if __name__ == "__main__":
    main()
