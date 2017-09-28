/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/net.h"
#include "caffe2/core/net_simple.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(
    NetRegistry,
    NetBase,
    const std::shared_ptr<const NetDef>&,
    Workspace*);

NetBase::NetBase(
    const std::shared_ptr<const NetDef>& def,
    Workspace* /* unused */)
    : external_input_(
          def->external_input().begin(),
          def->external_input().end()),
      external_output_(
          def->external_output().begin(),
          def->external_output().end()),
      name_(def->name()) {
  // Check that node_name is empty for all ops
  for (const OperatorDef& op : def->op()) {
    if (op.has_device_option()) {
      CAFFE_ENFORCE(
          !op.device_option().has_node_name(),
          "node_name must be empty for all operators at execution time.");
    }
  }

  // Go through the operators and make sure that blobs are correctly made.
  std::set<string> known_blobs(
      external_input_.begin(), external_input_.end());
  std::set<string> remaining_output(
      external_output_.begin(), external_output_.end());
  for (const auto& blob : known_blobs) {
    remaining_output.erase(blob);
  }
  for (const OperatorDef& op : def->op()) {
    for (const string& in : op.input()) {
      if (!known_blobs.count(in)) {
        if (external_input_.size()) {
          CAFFE_THROW(
              "op ",
              op.type(),
              ": Source for input ",
              in,
              " is unknown for net ",
              def->name(),
              ", operator ",
              ProtoDebugString(op));
        } else {
          // If we are not declaring input and output, we will simply VLOG it
          // for debugging purposes.
          VLOG(1) << "op " << op.type() << ": input " << in << " is unknown.";
        }
      }
    }
    for (const string& out : op.output()) {
      known_blobs.insert(out);
      remaining_output.erase(out);
    }
  }
  // Finally, check if all declared outputs are being created.
  CAFFE_ENFORCE(
      remaining_output.size() == 0,
      "Some of the blobs are declared as output but never produced by the "
      "net ",
      def->name(),
      ", the first one is ",
      *remaining_output.begin());
}

static NetObserverCreator GlobalNetObserverCreator = [](NetBase* net) {
  // A no-op ObserverBase<NetBase> observer
  return std::unique_ptr<NetObserver>(new NetObserver(net));
};

void SetGlobalNetObserverCreator(NetObserverCreator creator) {
  GlobalNetObserverCreator = creator;
  VLOG(1) << "Have set custom GlobalNetObserverCreator";
}

unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, ws);
}

unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  // In default, we will return a simple network that just runs all operators
  // sequentially.
  unique_ptr<NetBase> net;
  if (!net_def->has_type()) {
    net = std::unique_ptr<NetBase>(new SimpleNet(net_def, ws));
  } else {
    net = NetRegistry()->Create(net_def->type(), net_def, ws);
  }
  VLOG(1) << "Adding a global observer to a net";
  if (net) {
    net->SetObserver(GlobalNetObserverCreator(net.get()));
  }
  return net;
}

}  // namespace caffe2
