#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  status=${1:-64}
  cat >&2 <<EOF
usage: $0 {k8|k16} [options] [OmegaConf override ...]

options:
  --output DIR                 training output directory
  --teacher-checkpoint FILE    Boltz-2 teacher checkpoint
  --devices N                  GPUs per node
  --nodes N                    number of nodes
  --accumulate N               gradient accumulation batches
  --max-steps N                optimizer steps
  --resume FILE                resume checkpoint
  -h, --help                   show this message
EOF
  exit "${status}"
}

[[ $# -ge 1 ]] || usage
policy=$1
shift
case "${policy}" in
  k8) config=scripts/train/configs/mscd_boltz2_k8_rho40.yaml ;;
  k16) config=scripts/train/configs/mscd_boltz2_k16_rho40_normalign.yaml ;;
  *) usage ;;
esac

overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || usage
      export BOLTZ_MSCD_OUTPUT=$2
      shift 2
      ;;
    --teacher-checkpoint)
      [[ $# -ge 2 ]] || usage
      export BOLTZ2_TEACHER_CKPT=$2
      shift 2
      ;;
    --devices|--nodes|--accumulate|--max-steps)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || usage
      case "$1" in
        --devices) key=trainer.devices ;;
        --nodes) key=trainer.num_nodes ;;
        --accumulate) key=trainer.accumulate_grad_batches ;;
        --max-steps) key=trainer.max_steps ;;
      esac
      overrides+=("${key}=$2")
      shift 2
      ;;
    --resume)
      [[ $# -ge 2 ]] || usage
      overrides+=("resume=$2")
      shift 2
      ;;
    -h|--help)
      usage 0
      ;;
    --)
      shift
      overrides+=("$@")
      break
      ;;
    *=*)
      overrides+=("$1")
      shift
      ;;
    *)
      usage
      ;;
  esac
done

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
exec python scripts/train/train.py "${config}" "${overrides[@]}"
