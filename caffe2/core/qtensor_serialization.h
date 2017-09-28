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

#ifndef CAFFE2_CORE_QTENSOR_SERIALIZATION_H_
#define CAFFE2_CORE_QTENSOR_SERIALIZATION_H_

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/qtensor.h"

namespace caffe2 {

constexpr auto kQTensorBlobQType = "QTensor";

template <class Context>
class QTensorSerializer : public BlobSerializerBase {
 public:
  QTensorSerializer() : context_() {}
  ~QTensorSerializer() {}
  /**
   * Serializes a Blob. Note that this blob has to contain QTensor<Context>.
   */
  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override;

 private:
  Context context_;
};

template <class Context>
class QTensorDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
  void Deserialize(const QTensorProto& proto, QTensor<Context>* tensor);
};

template <class Context>
void QTensorSerializer<Context>::Serialize(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  const auto& qtensor = blob.template Get<QTensor<Context>>();
  BlobProto blob_proto;
  blob_proto.set_name(name);
  blob_proto.set_type(kQTensorBlobQType);
  QTensorProto& proto = *blob_proto.mutable_qtensor();
  proto.set_name(name);
  for (int i = 0; i < qtensor.ndim(); ++i) {
    proto.add_dims(qtensor.dim32(i));
  }
  proto.set_precision(qtensor.precision());
  proto.set_scale(qtensor.scale());
  proto.set_bias(qtensor.bias());
  proto.set_is_signed(qtensor.is_signed());
  detail::CopyToProtoWithCast(
      qtensor.nbytes(), qtensor.data(), proto.mutable_data(), &this->context_);
  acceptor(name, blob_proto.SerializeAsString());
}

template <class Context>
void QTensorDeserializer<Context>::Deserialize(
    const BlobProto& blob_proto,
    Blob* blob) {
  Deserialize(blob_proto.qtensor(), blob->GetMutable<QTensor<Context>>());
}

template <class Context>
void QTensorDeserializer<Context>::Deserialize(
    const QTensorProto& proto,
    QTensor<Context>* qtensor) {
  Context context{};
  vector<int> dims;
  for (const int d : proto.dims()) {
    dims.push_back(d);
  }
  qtensor->Resize(dims);
  qtensor->SetPrecision(proto.precision());
  qtensor->SetScale(proto.scale());
  qtensor->SetBias(proto.bias());
  qtensor->SetSigned(proto.is_signed());

  detail::CopyFromProtoWithCast(
      qtensor->nbytes(), proto.data(), qtensor->mutable_data(), &context);
}

} // namespace caffe2

#endif // CAFFE2_CORE_QTENSOR_SERIALIZATION_H_
