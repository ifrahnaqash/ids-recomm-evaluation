#!/usr/bin/env bash
# Package the per-run prediction arrays for deposit on Zenodo.
#
#   ./package_for_zenodo.sh /path/to/work-directory
#
# Produces, in ./zenodo_upload/ :
#   ids-recomm-predictions.tar.gz   the 1,500 prediction files
#   MANIFEST.csv                    one row per file, with checksum and size
#   README_ARCHIVE.md               what the archive contains and how to read it
#   CHECKSUMS.sha256                integrity check for the archive itself
#
# The source datasets are NOT included. Their licences rest with their original
# providers, and the archive contains model outputs only.

set -euo pipefail

WORK="${1:-}"
if [[ -z "$WORK" || ! -d "$WORK/predictions" ]]; then
    echo "usage: $0 /path/to/work-directory" >&2
    echo "  must contain a predictions/ subdirectory" >&2
    exit 1
fi

OUT="zenodo_upload"
mkdir -p "$OUT"

echo "==> inventory"
N=$(find "$WORK/predictions" -name '*.npz' | wc -l)
SZ=$(du -sh "$WORK/predictions" | cut -f1)
echo "    $N prediction files, $SZ"

if [[ "$N" -ne 1500 ]]; then
    echo "    WARNING: expected 1500 files, found $N."
    echo "    Check the run completed before depositing."
fi

echo "==> building manifest"
{
  echo "filename,dataset,balance,features,model,seed,bytes,sha256"
  find "$WORK/predictions" -name '*.npz' | sort | while read -r f; do
      b=$(basename "$f" .npz)
      # tag format: dataset__balance__features__model__seedNN
      IFS='_' read -r -a _ <<< ""   # placeholder to keep shellcheck quiet
      ds=${b%%__*};   rest=${b#*__}
      bal=${rest%%__*}; rest=${rest#*__}
      feat=${rest%%__*}; rest=${rest#*__}
      mdl=${rest%%__*}; seed=${rest##*__seed}
      sz=$(stat -c%s "$f")
      sum=$(sha256sum "$f" | cut -d' ' -f1)
      echo "$b.npz,$ds,$bal,$feat,$mdl,$seed,$sz,$sum"
  done
} > "$OUT/MANIFEST.csv"
echo "    -> $OUT/MANIFEST.csv ($(($(wc -l < "$OUT/MANIFEST.csv") - 1)) rows)"

echo "==> writing archive README"
cat > "$OUT/README_ARCHIVE.md" <<'EOF'
# IDS-ReComm: per-run model predictions

Model outputs for all 1,500 experimental runs reported in:

> I. Sanober and R. N. Mir, "A Comprehensive Evaluation of IDS Datasets and Models
> with a Derived Framework for Informed Dataset Selection: IDS-ReComm,"
> Advances in Computing and Engineering.

Code, aggregated metrics and figures:
https://github.com/ifrah-sanober/ids-recomm-evaluation

## Contents

`ids-recomm-predictions.tar.gz` unpacks to `predictions/`, containing one
compressed NumPy archive per run, named:

    <dataset>__<balance>__<features>__<model>__seed<NN>.npz

for example `CIC-IDS2017__balanced__all__RF__seed42.npz`.

- dataset:  NSL-KDD, UNSW-NB15, CIC-IDS2017, HIKARI-2021, HIKARI-2021-noID
- balance:  imbalanced, balanced (SMOTE applied to the training split only)
- features: all, top20, top10 (nested selections from recursive feature
            elimination fitted on training data only)
- model:    DT, RF, GNB, KNN, SVM, GRU, LSTM, MLP, SimpleRNN, CNN
- seed:     42 to 46

Each file holds three arrays over the held-out test split, in row order:

    y_true   int8      ground-truth label, 0 benign and 1 attack
    y_pred   int8      predicted label at a 0.5 decision threshold
    y_prob   float32   predicted probability of the attack class

## Reading a file

```python
import numpy as np
z = np.load("predictions/CIC-IDS2017__balanced__all__RF__seed42.npz")
y_true, y_pred, y_prob = z["y_true"], z["y_pred"], z["y_prob"]

from sklearn.metrics import f1_score, average_precision_score
f1_score(y_true, y_pred)                  # positive class
average_precision_score(y_true, y_prob)   # PR-AUC
```

Any metric in the paper can be recomputed from these arrays without retraining,
including metrics the paper does not report.

## What is not here

The four source datasets are not redistributed. Their licences rest with their
original providers, and they should be obtained from:

- NSL-KDD      https://www.unb.ca/cic/datasets/nsl.html
- UNSW-NB15    https://research.unsw.edu.au/projects/unsw-nb15-dataset
- CIC-IDS2017  https://www.unb.ca/cic/datasets/ids-2017.html
- HIKARI-2021  https://zenodo.org/record/6463389

The cleaned and split inputs are reproducible from those sources by running
`prepare.py` in the code repository, which is deterministic given a seed.

## Integrity

`MANIFEST.csv` lists every file with its size and SHA-256 checksum.
`CHECKSUMS.sha256` covers the archive itself:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0).
EOF

echo "==> creating archive (this takes a few minutes)"
tar -czf "$OUT/ids-recomm-predictions.tar.gz" -C "$WORK" predictions
ASZ=$(du -h "$OUT/ids-recomm-predictions.tar.gz" | cut -f1)
echo "    -> $OUT/ids-recomm-predictions.tar.gz ($ASZ)"

echo "==> checksums"
( cd "$OUT" && sha256sum ids-recomm-predictions.tar.gz MANIFEST.csv > CHECKSUMS.sha256 )
cat "$OUT/CHECKSUMS.sha256"

echo ""
echo "==> ready to upload"
echo "    Files in $OUT/:"
ls -lh "$OUT" | tail -n +2 | awk '{printf "      %-38s %s\n", $9, $5}'
echo ""
echo "    Zenodo accepts up to 50 GB per record, so a single archive is fine."
echo "    Upload all four files, not just the tarball."
