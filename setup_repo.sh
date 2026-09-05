#!/usr/bin/env bash
# Assemble and initialise the release repository.
#
#   ./setup_repo.sh /path/to/ids_rerun/work
#
# The argument is the WORK directory from config.py, containing results/,
# predictions/ and cache/. Run this from the repository root.

set -euo pipefail

WORK="${1:-}"
if [[ -z "$WORK" || ! -d "$WORK" ]]; then
    echo "usage: $0 /path/to/work-directory" >&2
    echo "  (the WORK path set in ids_recomm/config.py)" >&2
    exit 1
fi

echo "==> collecting results from $WORK"

mkdir -p results figures

# --- metrics: small enough for git -----------------------------------------
if [[ -d "$WORK/results" ]]; then
    n=$(find "$WORK/results" -name '*.json' | wc -l)
    echo "    $n per-run JSON files"
    mkdir -p results/per_run
    cp "$WORK"/results/*.json results/per_run/ 2>/dev/null || true
    # one archive rather than 1,500 loose files in the tree
    tar -czf results/per_run_metrics.tar.gz -C results per_run
    rm -rf results/per_run
    echo "    -> results/per_run_metrics.tar.gz ($(du -h results/per_run_metrics.tar.gz | cut -f1))"
fi

for f in results_tidy.csv results_summary.csv audit_report.txt; do
    [[ -f "$f" ]] && cp "$f" results/ && echo "    -> results/$f"
done
for f in stats_*.txt stats_*.json; do
    [[ -f "$f" ]] && cp "$f" results/
done

# --- dataset summaries ------------------------------------------------------
if compgen -G "$WORK/cache/clean_*_summary.json" > /dev/null; then
    mkdir -p results/dataset_summaries
    cp "$WORK"/cache/clean_*_summary.json results/dataset_summaries/
    cp "$WORK"/cache/features_*.json results/dataset_summaries/ 2>/dev/null || true
    echo "    -> results/dataset_summaries/"
fi

# --- predictions: too large for git, stage for Zenodo -----------------------
if [[ -d "$WORK/predictions" ]]; then
    n=$(find "$WORK/predictions" -name '*.npz' | wc -l)
    sz=$(du -sh "$WORK/predictions" | cut -f1)
    echo ""
    echo "==> $n prediction files, $sz total"
    echo "    These are NOT committed to git. To prepare the Zenodo upload:"
    echo ""
    echo "      tar -czf ids-recomm-predictions.tar.gz -C $WORK predictions"
    echo ""
    echo "    Upload that archive to https://zenodo.org, then put the DOI in"
    echo "    README.md and in the paper's Data Availability statement."
fi

# --- size check -------------------------------------------------------------
echo ""
echo "==> repository size check"
total=$(du -sh --exclude=.git . | cut -f1)
echo "    working tree: $total"
big=$(find . -path ./.git -prune -o -type f -size +50M -print 2>/dev/null || true)
if [[ -n "$big" ]]; then
    echo "    WARNING, files over 50 MB (GitHub hard limit is 100 MB):"
    echo "$big" | sed 's/^/      /'
fi

# --- git --------------------------------------------------------------------
if [[ ! -d .git ]]; then
    echo ""
    echo "==> initialising git"
    git init -q
    git add -A
    git commit -q -m "IDS-ReComm evaluation pipeline and results (1,500 runs)"
    echo "    committed. Next:"
    echo "      git branch -M main"
    echo "      git remote add origin git@github.com:<user>/ids-recomm-evaluation.git"
    echo "      git push -u origin main"
else
    echo ""
    echo "==> git already initialised; review with 'git status' before committing"
fi
