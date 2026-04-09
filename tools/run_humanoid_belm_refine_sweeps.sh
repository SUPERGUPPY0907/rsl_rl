#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSL_RL_ROOT="${RSL_RL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ISAACLAB_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)/Isaaclab_pr"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${ISAACLAB_ROOT_DEFAULT}}"

DEFAULT_TASKS="Isaac-Humanoid-v0"
TASKS="${TASKS:-${TASK:-${DEFAULT_TASKS}}}"
AGENT="${AGENT:-belm_genpo}"
SEEDS="${SEEDS:-42}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
HEADLESS="${HEADLESS:-1}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-belm_refine}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${ISAACLAB_ROOT}/logs/belm_refine_sweeps}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
SLOT_POLL_INTERVAL="${SLOT_POLL_INTERVAL:-10}"

CENTER_LAG="${CENTER_LAG:-0.75}"
BASE_B="${BASE_B:-${CENTER_LAG}}"
BASE_A="${BASE_A:-0.25}"
BASE_EPS="${BASE_EPS:-1.0}"

LAG_SWEEP="${LAG_SWEEP:-0.65,0.70,0.75,0.80,0.85}"
A_SWEEP="${A_SWEEP:-0.15,0.20,0.25,0.30,0.35}"
B_SWEEP="${B_SWEEP:-0.65,0.70,0.75,0.80,0.85}"
EPS_SWEEP="${EPS_SWEEP:-0.50,0.75,1.00,1.25,1.50}"

TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

usage() {
    cat <<EOF
Usage:
  bash ${0##*/} [--skip-install]

Environment overrides:
  ISAACLAB_ROOT     IsaacLab root directory. Default: ${ISAACLAB_ROOT_DEFAULT}
  RSL_RL_ROOT       Local rsl_rl checkout. Default: inferred from this script.
  TASKS             Comma-separated tasks. Default: ${DEFAULT_TASKS}
  TASK              Backward-compatible single-task override.
  SEEDS             Comma-separated seeds. Default: 42
  GPU_IDS           Comma-separated GPU ids. Default: 0,1,2,3,4,5,6,7
  HEADLESS          1 to add --headless, 0 otherwise. Default: 1
  NUM_ENVS          Optional --num_envs override.
  MAX_ITERATIONS    Optional --max_iterations override.
  SAVE_INTERVAL     Optional Hydra override for agent.save_interval.
  EXPERIMENT_PREFIX Experiment name prefix. Default: belm_refine
  LAUNCH_LOG_DIR    Per-run log directory. Default: logs/belm_refine_sweeps
  INSTALL_EDITABLE  1 to install local rsl_rl into IsaacLab env before running.
  SLOT_POLL_INTERVAL
                    Seconds to wait before checking for a free GPU slot. Default: 10

Refinement defaults around lag_coeff=0.75:
  CENTER_LAG        Default: 0.75
  BASE_A            Default: 0.25
  BASE_B            Default: 0.75
  BASE_EPS          Default: 1.0
  LAG_SWEEP         Default: 0.65,0.70,0.75,0.80,0.85
  A_SWEEP           Default: 0.15,0.20,0.25,0.30,0.35
  B_SWEEP           Default: 0.65,0.70,0.75,0.80,0.85
  EPS_SWEEP         Default: 0.50,0.75,1.00,1.25,1.50

Examples:
  bash ${0##*/}
  TASKS=Isaac-Humanoid-v0 SEEDS=1,2 GPU_IDS=0,1 bash ${0##*/}
  A_SWEEP=0.20,0.25,0.30 B_SWEEP=0.70,0.75,0.80 EPS_SWEEP=0.75,1.0,1.25 bash ${0##*/}
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--skip-install" ]]; then
    INSTALL_EDITABLE=0
fi

csv_to_array() {
    local csv="$1"
    local -n out_ref="$2"
    out_ref=()
    [[ -z "${csv}" ]] && return 0
    IFS=',' read -r -a out_ref <<< "${csv}"
}

trim_value() {
    local value="$1"
    value="${value// /}"
    echo "${value}"
}

sanitize_value() {
    local value
    value="$(trim_value "$1")"
    value="${value//-/neg}"
    value="${value//./p}"
    echo "${value}"
}

same_value() {
    [[ "$(sanitize_value "$1")" == "$(sanitize_value "$2")" ]]
}

task_slug() {
    local task="$1"
    case "${task}" in
        Isaac-Velocity-Rough-H1-v0) echo "h1_rough" ;;
        Isaac-Velocity-Rough-G1-v0) echo "g1_rough" ;;
        Isaac-Tracking-LocoManip-Digit-v0) echo "digit_loco_manip" ;;
        Isaac-Open-Drawer-Franka-v0) echo "franka_open_drawer" ;;
        Isaac-Humanoid-v0) echo "humanoid" ;;
        *)
            echo "Unsupported task for BELM refinement sweep: ${task}" >&2
            exit 1
            ;;
    esac
}

task_wandb_project() {
    local task="$1"
    case "${task}" in
        Isaac-Velocity-Rough-H1-v0) echo "belm_h1_rough" ;;
        Isaac-Velocity-Rough-G1-v0) echo "belm_g1_rough" ;;
        Isaac-Tracking-LocoManip-Digit-v0) echo "belm_digit_loco_manip" ;;
        Isaac-Open-Drawer-Franka-v0) echo "belm_franka_open_drawer" ;;
        Isaac-Humanoid-v0) echo "belm_humanoid" ;;
        *)
            echo "Unsupported task for BELM refinement sweep: ${task}" >&2
            exit 1
            ;;
    esac
}

build_experiment_name() {
    local task_label="$1"
    local variant="$2"
    local seed="$3"
    echo "${EXPERIMENT_PREFIX}_${task_label}_${variant}_seed${seed}"
}

declare -a gpu_ids=()
declare -a slot_pids=()
declare -a slot_experiments=()
declare -a slot_status_files=()
declare -i failed_jobs=0

initialize_gpu_ids() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required to determine available GPUs." >&2
        exit 1
    fi

    if [[ -n "${GPU_IDS}" ]]; then
        csv_to_array "${GPU_IDS}" gpu_ids
    else
        mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    fi

    if [[ ${#gpu_ids[@]} -eq 0 ]]; then
        echo "No GPUs configured for the sweep." >&2
        exit 1
    fi

    local i
    for i in "${!gpu_ids[@]}"; do
        gpu_ids[$i]="$(trim_value "${gpu_ids[$i]}")"
        slot_pids[$i]=""
        slot_experiments[$i]=""
        slot_status_files[$i]=""
    done
}

terminate_running_jobs() {
    local pid
    for pid in "${slot_pids[@]}"; do
        [[ -z "${pid}" ]] && continue
        kill "${pid}" 2>/dev/null || true
    done
}

reap_finished_jobs() {
    local i pid status_file status gpu_id experiment_name

    for i in "${!gpu_ids[@]}"; do
        pid="${slot_pids[$i]}"
        status_file="${slot_status_files[$i]}"

        [[ -z "${pid}" || -z "${status_file}" || ! -f "${status_file}" ]] && continue

        status="$(<"${status_file}")"
        wait "${pid}" 2>/dev/null || true

        gpu_id="${gpu_ids[$i]}"
        experiment_name="${slot_experiments[$i]}"
        echo "[INFO] Experiment '${experiment_name}' on GPU ${gpu_id} finished with status ${status}"

        if [[ "${status}" -ne 0 ]]; then
            failed_jobs+=1
        fi

        rm -f "${status_file}"
        slot_pids[$i]=""
        slot_experiments[$i]=""
        slot_status_files[$i]=""
    done
}

wait_for_free_slot() {
    local i
    while true; do
        reap_finished_jobs
        for i in "${!gpu_ids[@]}"; do
            if [[ -z "${slot_pids[$i]}" ]]; then
                echo "${i}"
                return 0
            fi
        done
        sleep "${SLOT_POLL_INTERVAL}"
    done
}

launch_run() {
    local slot_index="$1"
    local task="$2"
    local experiment_name="$3"
    local wandb_project="$4"
    local label="$5"
    shift 5

    local gpu_id="${gpu_ids[$slot_index]}"
    local log_file="${LAUNCH_LOG_DIR}/${experiment_name}.log"
    local status_file="${LAUNCH_LOG_DIR}/${experiment_name}.status"

    local -a run_args=(
        --task "${task}"
        --agent "${AGENT}"
        --device "cuda:${gpu_id}"
    )

    if [[ -n "${MAX_ITERATIONS}" ]]; then
        run_args+=(--max_iterations "${MAX_ITERATIONS}")
    fi
    if [[ "${HEADLESS}" == "1" ]]; then
        run_args+=(--headless)
    fi
    if [[ -n "${NUM_ENVS}" ]]; then
        run_args+=(--num_envs "${NUM_ENVS}")
    fi

    local -a hydra_args=(
        "agent.experiment_name=${experiment_name}"
        "agent.logger=wandb"
        "agent.wandb_project=${wandb_project}"
        "$@"
    )

    if [[ -n "${SAVE_INTERVAL}" ]]; then
        hydra_args+=("agent.save_interval=${SAVE_INTERVAL}")
    fi

    rm -f "${status_file}"

    echo
    echo "==== Launching ${task} :: ${label} on GPU ${gpu_id} ===="
    echo "Experiment name: ${experiment_name}"
    echo "wandb_project: ${wandb_project}"
    printf 'Overrides:'
    printf ' %q' "${hydra_args[@]}"
    printf '\n'

    (
        set +e
        cd "${ISAACLAB_ROOT}" || exit 1
        ./isaaclab.sh -p "${TRAIN_SCRIPT}" "${run_args[@]}" "${hydra_args[@]}"
        status=$?
        printf '%s\n' "${status}" > "${status_file}"
        exit "${status}"
    ) > "${log_file}" 2>&1 &

    slot_pids[$slot_index]="$!"
    slot_experiments[$slot_index]="${experiment_name}"
    slot_status_files[$slot_index]="${status_file}"

    echo "[INFO] Started '${experiment_name}' on GPU ${gpu_id} with pid ${slot_pids[$slot_index]}"
    echo "[INFO] Launcher log: ${log_file}"
}

schedule_run() {
    local slot_index
    slot_index="$(wait_for_free_slot)"
    launch_run "${slot_index}" "$@"
}

wait_for_all_jobs() {
    local slot_has_jobs
    while true; do
        reap_finished_jobs
        slot_has_jobs=0
        for pid in "${slot_pids[@]}"; do
            if [[ -n "${pid}" ]]; then
                slot_has_jobs=1
                break
            fi
        done
        [[ "${slot_has_jobs}" -eq 0 ]] && return 0
        sleep "${SLOT_POLL_INTERVAL}"
    done
}

schedule_untied_variant() {
    local task="$1"
    local task_label="$2"
    local wandb_project="$3"
    local seed="$4"
    local variant_prefix="$5"
    local a_value="$6"
    local b_value="$7"
    local eps_value="$8"

    local variant="${variant_prefix}_a$(sanitize_value "${a_value}")_b$(sanitize_value "${b_value}")_eps$(sanitize_value "${eps_value}")"
    schedule_run "${task}" "$(build_experiment_name "${task_label}" "${variant}" "${seed}")" "${wandb_project}" "${variant}" \
        "agent.seed=${seed}" \
        "agent.policy.lag_coeff=null" \
        "agent.policy.a_coeff=${a_value}" \
        "agent.policy.b_coeff=${b_value}" \
        "agent.policy.eps_coeff=${eps_value}"
}

schedule_tied_variant() {
    local task="$1"
    local task_label="$2"
    local wandb_project="$3"
    local seed="$4"
    local lag_value="$5"

    local variant="tied_lag$(sanitize_value "${lag_value}")"
    schedule_run "${task}" "$(build_experiment_name "${task_label}" "${variant}" "${seed}")" "${wandb_project}" "${variant}" \
        "agent.seed=${seed}" \
        "agent.policy.a_coeff=null" \
        "agent.policy.b_coeff=null" \
        "agent.policy.eps_coeff=null" \
        "agent.policy.lag_coeff=${lag_value}"
}

trap 'terminate_running_jobs; exit 130' INT TERM

declare -a tasks=()
declare -a seeds=()
declare -a lag_values=()
declare -a a_values=()
declare -a b_values=()
declare -a eps_values=()

csv_to_array "${TASKS}" tasks
csv_to_array "${SEEDS}" seeds
csv_to_array "${LAG_SWEEP}" lag_values
csv_to_array "${A_SWEEP}" a_values
csv_to_array "${B_SWEEP}" b_values
csv_to_array "${EPS_SWEEP}" eps_values
initialize_gpu_ids

mkdir -p "${LAUNCH_LOG_DIR}"

echo "IsaacLab root: ${ISAACLAB_ROOT}"
echo "rsl_rl root: ${RSL_RL_ROOT}"
echo "Tasks: ${TASKS}"
echo "Agent: ${AGENT}"
echo "Seeds: ${SEEDS}"
echo "GPU ids: ${gpu_ids[*]}"
echo "Max iterations override: ${MAX_ITERATIONS:-<task cfg default>}"
echo "Save interval override: ${SAVE_INTERVAL:-<task cfg default>}"
echo "Num envs override: ${NUM_ENVS:-<task cfg default>}"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "Launcher log dir: ${LAUNCH_LOG_DIR}"
echo "Center lag: ${CENTER_LAG}"
echo "Baseline untied coeffs: a=${BASE_A}, b=${BASE_B}, eps=${BASE_EPS}"
echo "Lag sweep: ${LAG_SWEEP:-<disabled>}"
echo "A sweep: ${A_SWEEP:-<disabled>}"
echo "B sweep: ${B_SWEEP:-<disabled>}"
echo "EPS sweep: ${EPS_SWEEP:-<disabled>}"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
    echo
    echo "==== Installing local rsl_rl into IsaacLab Python environment ===="
    (
        cd "${ISAACLAB_ROOT}"
        ./isaaclab.sh -p -m pip install -e "${RSL_RL_ROOT}"
    )
fi

for task in "${tasks[@]}"; do
    task="$(trim_value "${task}")"
    task_label="$(task_slug "${task}")"
    wandb_project="$(task_wandb_project "${task}")"

    echo
    echo "==== Starting refinement sweep for ${task} (${task_label}) ===="

    for seed in "${seeds[@]}"; do
        seed="$(trim_value "${seed}")"

        schedule_untied_variant "${task}" "${task_label}" "${wandb_project}" "${seed}" "local_base" "${BASE_A}" "${BASE_B}" "${BASE_EPS}"

        for lag_value in "${lag_values[@]}"; do
            lag_value="$(trim_value "${lag_value}")"
            [[ -z "${lag_value}" ]] && continue
            schedule_tied_variant "${task}" "${task_label}" "${wandb_project}" "${seed}" "${lag_value}"
        done

        for a_value in "${a_values[@]}"; do
            a_value="$(trim_value "${a_value}")"
            [[ -z "${a_value}" ]] && continue
            same_value "${a_value}" "${BASE_A}" && continue
            schedule_untied_variant "${task}" "${task_label}" "${wandb_project}" "${seed}" "a_refine" "${a_value}" "${BASE_B}" "${BASE_EPS}"
        done

        for b_value in "${b_values[@]}"; do
            b_value="$(trim_value "${b_value}")"
            [[ -z "${b_value}" ]] && continue
            same_value "${b_value}" "${BASE_B}" && continue
            schedule_untied_variant "${task}" "${task_label}" "${wandb_project}" "${seed}" "b_refine" "${BASE_A}" "${b_value}" "${BASE_EPS}"
        done

        for eps_value in "${eps_values[@]}"; do
            eps_value="$(trim_value "${eps_value}")"
            [[ -z "${eps_value}" ]] && continue
            same_value "${eps_value}" "${BASE_EPS}" && continue
            schedule_untied_variant "${task}" "${task_label}" "${wandb_project}" "${seed}" "eps_refine" "${BASE_A}" "${BASE_B}" "${eps_value}"
        done
    done
done

wait_for_all_jobs

echo
if [[ ${failed_jobs} -ne 0 ]]; then
    echo "BELM refinement sweep finished with ${failed_jobs} failed job(s)." >&2
    exit 1
fi
echo "BELM refinement sweep finished successfully."
