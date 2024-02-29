#!/bin/bash

ENV_CONFIG_PATH="./configs/env_machine.yml"
EXPT_CONFIG_PATH="./configs/expts/pokemon_all.yml"

PYTHON_CMD_DISCOVERY="python -W ignore discovering.py --config_file_env $ENV_CONFIG_PATH --config_file_expt $EXPT_CONFIG_PATH"
PYTHON_CMD_EVAL="python -W ignore grouping.py --config_file_env $ENV_CONFIG_PATH --config_file_expt $EXPT_CONFIG_PATH"

# Core function to execute commands
run_cmds() {
    local num_categories=$1
    $PYTHON_CMD_EVAL --alpha 0.7 --N_tta 10 --num_per_category ${num_categories} --num_runs 10
}

run_cmds "3"