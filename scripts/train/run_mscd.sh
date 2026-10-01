#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 {k8|k16} [OmegaConf override ...]" >&2
  exit 64
}

[[ $# -ge 1 ]] || usage
policy=$1
shift
case "${policy}" in
  k8) config=scripts/train/configs/mscd_boltz2_k8_rho40.yaml ;;
  k16) config=scripts/train/configs/mscd_boltz2_k16_rho40_normalign.yaml ;;
  *) usage ;;
esac

required=(
  BOLTZ_MSCD_OUTPUT
  BOLTZ2_TEACHER_CKPT
  BOLTZ_RCSB_TARGET_DIR
  BOLTZ_RCSB_MSA_DIR
  BOLTZ_RCSB_TEMPLATE_DIR
  BOLTZ_AFDB_TARGET_DIR
  BOLTZ_AFDB_MSA_DIR
  BOLTZ_AFDB_PACK_INDEX
  BOLTZ_AFDB_SAMPLES_PATH
  BOLTZ_CCD_MOL_DIR
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "missing required environment variable: ${name}" >&2
    exit 64
  }
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python scripts/train/train.py "${config}" "$@"
