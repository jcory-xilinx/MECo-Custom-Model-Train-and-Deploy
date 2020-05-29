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

# Set working directory to toplevel
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
pushd $script_dir/..

#####################################
# Run MECo ML Compiler Initialization
# (DO NOT MODIFY THIS SCRIPT)
source ./scripts/_run_ml_init.sh

#####################################

if [[ ! -z $all ]]; then
  unset all
  cleanup=1
  ./$0 -r -c 14
  ./$0 -s -c 14
  ./$0 -r -c 98
  ./$0 -s -c 98
  ./$0 -r -c 336
  ./$0 -s -c 336
  ./$0 -r -c 392
  ./$0 -s -c 392
#  ./$0 -r -c 14 --mrt
#  ./$0 -s -c 14 --mrt
  exit 0
fi

if [[ ! -z $all_hw ]]; then
  unset all_hw
  if [[ -z $pcie ]]; then
    ./$0 -r -c 14
  else
    ./$0 -r -c 14 --pcie
  fi

  exit 0
fi

if [[ -z $mode || -z $cores ]]; then
  echo "Mandatory arguments (--static/--retrain + --cores) not provided"
  print_usage; exit 1
fi

#HERE SPECIFY ALL THE SETTINGS THAT DEPEND ON THE NUMBER OF CORES OF M2_N
if [[ $cores -eq 14 ]]; then
  cardano_reports_dir="${COMPILER_ROOT}/griffin/utils/tv_gen/cardano_reports/m2_14core"
  last_timestep=t10
  if [[ -z $num_phy_weight_streams ]]; then
    num_phy_weight_streams="2"
  fi
  if [[ -z $num_phy_ifm_streams ]]; then
    num_phy_ifm_streams="14"
  fi
  aie_ofm_sv_depth_mult_16="False"
  batch_size="1"
elif [[ $cores -eq 98 ]]; then
  cardano_reports_dir="${COMPILER_ROOT}/griffin/utils/tv_gen/cardano_reports/m2_98core"
  last_timestep=t10
  if [[ -z $num_phy_weight_streams ]]; then
    num_phy_weight_streams="14"
  fi
  if [[ -z $num_phy_ifm_streams ]]; then
    num_phy_ifm_streams="14"
  fi
  aie_ofm_sv_depth_mult_16="False"
  batch_size="1"
elif [[ $cores -eq 336 ]]; then
  cardano_reports_dir="${COMPILER_ROOT}/griffin/utils/tv_gen/cardano_reports/m2_392core"
  last_timestep=t4
  if [[ -z $num_phy_weight_streams ]]; then
    num_phy_weight_streams="8"
  fi
  if [[ -z $num_phy_ifm_streams ]]; then
    num_phy_ifm_streams="84"
  fi
  aie_ofm_sv_depth_mult_16="False"
  batch_size="6"
elif [[ $cores -eq 392 ]]; then
  cardano_reports_dir="${COMPILER_ROOT}/griffin/utils/tv_gen/cardano_reports/m2_392core"
  last_timestep=t10
  if [[ -z $num_phy_weight_streams ]]; then
    num_phy_weight_streams="14"
  fi
  if [[ -z $num_phy_ifm_streams ]]; then
    num_phy_ifm_streams="56"
  fi
  aie_ofm_sv_depth_mult_16="False"
  batch_size="4"
else
  echo "${cores} cofiguration currently unsupported."
  exit 1
fi

if [[ $mode == $STATIC ]]; then
  echo "Running in static mode with ${cores} cores"
  mname="resnet-21/"
  qmname="calibrated_ckpt_int8"
  mdir="${mroot}/${mname}"
else
  echo "Running in retrain mode with ${cores} cores"
  mname="resnet-21/"
  qmname="retrained_ckpt_int8_wt_th"
  mdir="${mroot}/${mname}"
fi

if [[ -z $builddir ]]; then
  builddir="build/${mname}/${cores}core"
  if [[ ! -z $mrt ]]; then
    builddir="${builddir}_metropolis_runtime"
  fi
fi

in_graph="$mdir/resnet_21_pretrained.pb"
opt_graph="${builddir}/resnet_21_opt.pb"
quant_graph="${builddir}/resnet_21_infquant.pb"
input_node="input_tensor"
output_node="softmax_tensor"
input_shape="224,224,3"
wb="-8"
ab="-8"
lb="-16"
rb="8"
pb="8"
prb="8"

first_layer="resnet_model/conv2d/Conv2D"
last_layer="resnet_model/dense/MatMul"

conv_split_list="${builddir}/conv_split_list.txt"
layer_merge_list="${builddir}/layer_merge_list.txt"
left_shift_list="${builddir}/left_shift_list.txt"
griffin_output="${builddir}/griffin_output.json"
pl_clk_freq="312.5e6"
aie_clk_freq="1.25e9"
enable_merge="True"
cover_full_tensor_height="True"
cdno_packet_report="${cardano_reports_dir}/packet_switching_report.json"
cdno_dma_report="${cardano_reports_dir}/dma_lock_report.json"
enable_implementation_mode="True"
model_curr_aie_implementation="False"
enable_layer_boundary_pipeline="True"
enable_FC_concurrency="True"
tensor_cache_size="1835008"
cdno_sync_buffer_report="${cardano_reports_dir}/sync_buffer_address.json"
tiling=True

if [[ ! -z $mrt ]]; then
  dump_partitions=True
  subgraph_config=${mdir}/subgraph_config.json
  mrt_option_griffin="--subgraph_config ${subgraph_config}"
  mrt_option_graffite=${mrt_option_griffin}
else
  dump_partitions=False
  subgraph_config=""
fi

# Allow Griffin begin and end nodes to be defined from the command line
if [[ -z ${griffin_begin_node_commandline+x} ]]; then
  griffin_begin_node=${first_layer}
else
  griffin_begin_node=${griffin_begin_node_commandline}
fi

if [[ -z ${griffin_end_node_commandline+x} ]]; then
  griffin_end_node=${last_layer}
else
  griffin_end_node=${griffin_end_node_commandline}
fi

if [[ -z ${start_time_step+x} ]]; then
  start_time_step="1"
fi

# Graffite
graffite_no_weights="False"
graffite_hw_config="hwconfig_m2_${cores}.json"

if [[ -z $pcie ]]; then
  #####################################
  # Run MECo ML Compiler Toolchain
  # (DO NOT MODIFY THIS SCRIPT)
  mkdir -p ${builddir}
  source ./scripts/_run_ml_compiler.sh $last_timestep
  #####################################

  if [[ ! -z $cleanup ]]; then
    echo "Deleting model files and checkpoints"
    /bin/rm -rf ${mdir}/*
    /bin/rm -f ./models/graffitist_ckpt/${qmname}/${mname}/*
    /bin/rm -f ./data/*.tfrecord
  fi

else # PCIE case
  echo -e "\e[31mWarning: skipping model compilation step, make sure the mem files aren't stale\e[0m"

  cd ./meco/granite
  source setup.sh > /dev/null
  cmake . > /dev/null
  cmake --build . > /dev/null

  builddir="${repo_path}/build/${mname}/${cores}core"
  golddir="${repo_path}/tests/golden/${mname}/${cores}core"
  ifm="ddr_ifm_img"
  ofm="ddr_ofm_img"
  log="hw_test_log_img"
  granite="${repo_path}/meco/granite/granite_slim.exe"
  pass=" \e[32mPASS\e[0m"
  fail=" \e[31mFAIL\e[0m"
  if [[ $cores -eq 14 ]]; then
    xclbin="${repo_path}/setup/xclbin/cnn_${cores}core_${platform}_ea2.xclbin"
  else
    xclbin="${repo_path}/setup/xclbin/cnn_${cores}core_${platform}_${network}.xclbin"
  fi

  echo -e "\e[1mTEST RESULTS:\e[21m"

  for i in {0..15}; do
    testName="RESNET_V1P5_50 IMAGE ${i}: "
    result=`${granite} ${xclbin} ${builddir} "${golddir}/${ifm}_${i}.mem" "${golddir}/${ofm}_${i}.mem" |& tee "${builddir}/${log}_${i}.txt" | grep "Matched!"`
    if [[ $result = "Outputs Matched!" ]]; then
      echo -e "${testName}${pass}"
    else
      echo -e "${testName}${fail}"
    fi
  done

fi
