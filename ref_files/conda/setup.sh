#!/usr/bin/env bash

# © Copyright (C) 2016-2017 Xilinx, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
export script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
pushd $script_dir/..

ENV_NAME=tf1.14-gpu
echo "Setting up environment $ENV_NAME"

conda create -y -n ${ENV_NAME} pip python=3.6 --file ./conda/conda_requirements.txt -c defaults -c conda-forge

# Fix for conda acivate issues from bash script
# https://github.com/conda/conda/issues/7980
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

pip install -r ./conda/pip_requirements.txt
pip install -e ./meco/graffitist
pip install -e ./
