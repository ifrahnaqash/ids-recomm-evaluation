"""
Stage 1 + 2.

    python prepare.py --dataset NSL-KDD --stage clean
    python prepare.py --dataset NSL-KDD --stage prep --seed 42

Stage `clean` reads the raw CSVs, cleans them, and writes one parquet.
Stage `prep` does the split and fits encoder / correlation filter / RFE /
scalers on training data only, then persists them.

Both stages skip work that has already been done, so a failed array job can be
resubmitted without redoing everything.
"""
import argparse
import json
import time

import config as C
import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASET_NAMES)
    ap.add_argument("--stage", required=True, choices=["clean", "prep"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    t0 = time.time()

    if a.stage == "clean":
        out = D.cache_path(a.dataset)
        if out.exists() and not a.force:
            print(f"[skip] {out} exists")
            return
        df = D.build_clean_frame(a.dataset)
        df.to_parquet(out, index=False)
        pos = float(df["__label__"].mean())
        print(f"[clean] {a.dataset}: {len(df):,} rows x {df.shape[1]-1} raw features, "
              f"attack rate {pos:.4f}  ({time.time()-t0:.1f}s)")
        (C.CACHE_DIR / f"clean_{a.dataset}_summary.json").write_text(json.dumps(
            dict(dataset=a.dataset, n_rows=int(len(df)),
                 n_raw_features=int(df.shape[1] - 1), attack_rate=pos,
                 derived_from=C.DATASETS[a.dataset].get("derive_from"),
                 **df.attrs.get("dup_stats", {})), indent=2))

    else:
        assert a.seed is not None, "--seed required for stage prep"
        out = D.prep_path(a.dataset, a.seed)
        if out.exists() and not a.force:
            print(f"[skip] {out} exists")
            return
        p = D.build_prep(a.dataset, a.seed)
        print(f"[prep] {a.dataset} seed={a.seed}: "
              f"train={p['n_train']:,} val={p['n_val']:,} test={p['n_test']:,}; "
              f"{len(p['kept_names'])} features after correlation filter "
              f"({len(p['corr_dropped'])} dropped, {len(p['constant_dropped'])} constant); "
              f"RFE top20={len(p['selections']['top20'])}, "
              f"top10={len(p['selections']['top10'])}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
