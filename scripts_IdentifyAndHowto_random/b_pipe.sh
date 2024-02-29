#!/bin/bash

ENV_CONFIG_PATH="./configs/env_machine.yml"
EXPT_CONFIG_PATH="./configs/expts/bird200_all.yml"

PYTHON_CMD_DISCOVERY="python -W ignore control_discovery.py --config_file_env $ENV_CONFIG_PATH --config_file_expt $EXPT_CONFIG_PATH"
PYTHON_CMD_EVAL="python -W ignore control_eval_grouping.py --config_file_env $ENV_CONFIG_PATH --config_file_expt $EXPT_CONFIG_PATH"

# Core function to execute commands
run_cmds() {
    local num_categories=$1
    $PYTHON_CMD_DISCOVERY --mode identify --num_per_category ${num_categories}
    $PYTHON_CMD_DISCOVERY --mode howto --num_per_category ${num_categories}
}

run_cmds "random"