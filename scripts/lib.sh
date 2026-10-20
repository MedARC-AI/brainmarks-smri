# Shared helpers for the dataset download scripts. Source from scripts/<name>/download.sh.

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${BRAINMARKS_SMRI_DATA:-$REPO/datasets}

die() { echo "error: $*" >&2; exit 1; }

# check_budget <bytes>: refuse to download if it would push datasets/ past 1 TB
# or the volume past 98% full.
check_budget() {
    local need=$1 used total vol_used
    used=$(du -sb "$DATA_ROOT" | cut -f1)
    (( used + need <= 10**12 )) || die "budget: datasets/ would exceed 1 TB"
    read -r total vol_used < <(df -B1 --output=size,used "$DATA_ROOT" | tail -1)
    (( (vol_used + need) * 100 <= total * 98 )) || die "budget: volume would exceed 98% full"
}

# write_manifest <name>: sha256 of every file under datasets/<name>/source/, written to
# datasets/<name>/tables/source.sha256 (check with `cd datasets/<name> && sha256sum -c tables/source.sha256`)
# and copied to scripts/<name>/tables/ so it's tracked in git. A `git diff` there flags any
# change in what a download produces.
write_manifest() {
    local dir=$DATA_ROOT/$1
    mkdir -p "$dir/tables" "$REPO/scripts/$1/tables"
    (cd "$dir" && find source -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$dir/tables/source.sha256"
    cp "$dir/tables/source.sha256" "$REPO/scripts/$1/tables/source.sha256"
}
