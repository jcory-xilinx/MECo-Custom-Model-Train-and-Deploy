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
import tensorflow as tf
import os

def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result

graph = tf.compat.v1.get_default_graph()
pb_filepath="./resnet_21_pretrained.pb"
saved_model_dir='./resnet_21/export_model/'

with tf.compat.v1.Session(graph=graph) as sess:
    all_subdirs = all_subdirs_of(saved_model_dir)
    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    #first load the saved model
    loader = tf.saved_model.load(sess,[tf.saved_model.SERVING],export_dir=latest_subdir)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    #this is the output node name for the resnet-21 model
    output_node_names = ['softmax_tensor']

    #convert the variables to constants in the graph
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

    #write out the output graph definition
    with tf.gfile.GFile(pb_filepath, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
