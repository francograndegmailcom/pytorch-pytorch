#pragma once

#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/utils/functional.h"

#include <ATen/ATen.h>

#include <memory>
#include <iostream>

namespace torch { namespace jit {

#define TH_FORALL_TYPES(_) \
_(DynamicType) \
_(TensorType) \
_(CompleteTensorType) \
_(TupleType) \
_(ListType) \
_(NumberType) \
_(FloatType) \
_(IntType) \
_(NoneType) \
_(StringType) \
_(GeneratorType) \

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  TH_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

struct Type;
using TypePtr = std::shared_ptr<Type>;

template<bool is_singleton, typename T>
struct cloneType {};

template<typename T>
struct cloneType<true, T> {
  std::shared_ptr<T> operator()(std::shared_ptr<T> ptr) const {
    return ptr;
  }
  std::shared_ptr<const T> operator()(std::shared_ptr<const T> ptr) const {
    return ptr;
  }
};

template<typename T>
struct cloneType<false, T> {
  std::shared_ptr<T> operator()(std::shared_ptr<const T> ptr) const {
    return std::make_shared<T>(*ptr);
  }
};

struct TORCH_API Type : std::enable_shared_from_this<Type> {
private:
  TypeKind kind_;

protected:
  Type(TypeKind kind)
    : kind_(kind) {}

public:
  virtual bool operator==(const Type& rhs) const = 0;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal
  virtual bool isSubtypeOf(const TypePtr rhs) const {
    return *this == *rhs;
  }
  // user-friendly form of the type, separate from
  // operator<< which is verbose and unambiguous
  virtual std::string str() const = 0;

  TypeKind kind() const {
    return kind_;
  }

  // Dynamically cast this object to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid.
  // NOTE: if the cast succeeds, but the casted kind is not the
  // run-time kind of the type, we also slice the structure, so
  // that assignments of those types to values don't accidentally
  // inherit more detailed information from subclasses.
  template<typename T>
  std::shared_ptr<T> cast() {
    auto r = std::dynamic_pointer_cast<T>(shared_from_this());
    if (!r || T::Kind == kind()) {
      return r;
    } else {
      return cloneType<T::is_singleton, T>{}(r);
    }
  }
  template<typename T>
  std::shared_ptr<const T> cast() const {
    auto r = std::dynamic_pointer_cast<const T>(shared_from_this());
    if (!r || T::Kind == kind()) {
      return r;
    } else {
      return cloneType<T::is_singleton, T>{}(r);
    }
  }
  template<typename T>
  std::shared_ptr<T> expect() {
    auto r = cast<T>();
    JIT_ASSERT(r);
    return r;
  }
  template<typename T>
  std::shared_ptr<const T> expect() const {
    auto r = cast<const T>();
    JIT_ASSERT(r);
    return r;
  }
  virtual ~Type() = default;
};

inline bool operator!=(const Type & lhs, const Type & rhs) {
  return !(lhs == rhs);
}

struct DynamicType;
using DynamicTypePtr = std::shared_ptr<DynamicType>;
// This node represents a single Tensor value, with an unknown shape.
struct TORCH_API DynamicType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static DynamicTypePtr create( T&& ... all ) {
    return DynamicTypePtr(new DynamicType( std::forward<T>(all)... ));
  }

  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Tensor";
  }
  static const TypeKind Kind = TypeKind::DynamicType;
  // global singleton
  static DynamicTypePtr get();
private:
  DynamicType()
  : Type(TypeKind::DynamicType) {}
};

struct TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
// This node represents a single Tensor value with a specific size
struct TORCH_API TensorType : public Type {
  static constexpr bool is_singleton = false;
  friend struct Type;
  static const TypeKind Kind = TypeKind::TensorType;

  template<typename ... T>
  static TensorTypePtr create( T&& ... all ) {
    return TensorTypePtr(new TensorType( std::forward<T>(all)... ));
  }

  at::ScalarType scalarType() const { return scalar_type_; }
  int device() const { return device_; }
  int dim() const { return dim_; }

  TensorTypePtr toScalarType(at::ScalarType type){
    auto t = TensorType::create(*this);
    t->scalar_type_ = type;
    return t;
  }
  TensorTypePtr withDim(int new_dim) {
    auto t = TensorType::create(*this);
    t->dim_ = new_dim;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if (rhs.kind() != TypeKind::TensorType)
      return false;
    auto rt = rhs.expect<TensorType>();
    return scalarType() == rt->scalarType() &&
           device() == rt->device() &&
           dim() == rt->dim();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if (rhs->kind() == TypeKind::DynamicType)
      return true;
    return rhs->kind() == TypeKind::TensorType && *this == *rhs;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }

protected:
  TensorType(const at::Tensor& tensor, TypeKind kind=TypeKind::TensorType)
    : TensorType(tensor.type().scalarType(), tensor.type().is_cuda() ? tensor.get_device() : -1, tensor.dim(), kind) {}
  TensorType(at::ScalarType scalar_type, int device, int dim, TypeKind kind=TypeKind::TensorType)
    : Type(kind)
    , scalar_type_(scalar_type)
    , device_(device)
    , dim_(dim) {}

  at::ScalarType scalar_type_;
  int device_;
  int dim_;
};

struct CompleteTensorType;
using CompleteTensorTypePtr = std::shared_ptr<CompleteTensorType>;
// This node represents a single Tensor value with a specific size
struct TORCH_API CompleteTensorType : public TensorType {
  static constexpr bool is_singleton = false;
  friend struct Type;
  template<typename ... T>
  static CompleteTensorTypePtr create( T&& ... all ) {
    return CompleteTensorTypePtr(new CompleteTensorType( std::forward<T>(all)... ));
  }

  // overloaded create variadic template argument as it could not distinguish initializer list
  static CompleteTensorTypePtr create(at::ScalarType scalar_type, int device, at::IntList sizes) {
    return CompleteTensorTypePtr(new CompleteTensorType(scalar_type, device, sizes));
  }
  static CompleteTensorTypePtr create(at::ScalarType scalar_type, int device, at::IntList sizes, at::IntList strides) {
    return CompleteTensorTypePtr(new CompleteTensorType(scalar_type, device, sizes, strides));
  }

  static const TypeKind Kind = TypeKind::CompleteTensorType;

  const std::vector<int64_t>& sizes() const { return sizes_; }
  const std::vector<int64_t>& strides() const { return strides_; }

  TypePtr withSizesStrides(at::IntList sizes, at::IntList strides) const {
    return CompleteTensorType::create(scalar_type_, device_, sizes, strides);
  }

  TypePtr withSizes(at::IntList sizes) const {
    return withSizesStrides(sizes, CompleteTensorType::contiguousStridesOf(sizes));
  }

  CompleteTensorTypePtr contiguous() const {
    auto t = CompleteTensorType::create(*this);
    t->strides_ = CompleteTensorType::contiguousStridesOf(sizes_);
    return t;
  }

  CompleteTensorTypePtr toScalarType(at::ScalarType type){
    auto t = CompleteTensorType::create(*this);
    t->scalar_type_ = type;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if(rhs.kind() != kind())
      return false;
    auto rt = rhs.expect<CompleteTensorType>();
    return scalarType() == rt->scalarType() &&
           sizes() == rt->sizes() &&
           strides() == rt->strides() &&
           device() == rt->device();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if (rhs->kind() == TypeKind::DynamicType)
      return true;
    if (rhs->kind() == TypeKind::TensorType)
      return *dynamic_cast<const TensorType*>(this) == *rhs;
    return *this == *rhs;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }
  bool numel() const {
    size_t prod = 1;
    for(auto s : sizes()) {
      prod *= s;
    }
    return prod;
  }
  static TypePtr fromNumberType(TypePtr typ);

private:
  CompleteTensorType(const at::Tensor& tensor)
    : TensorType(tensor, TypeKind::CompleteTensorType)
    , sizes_(tensor.sizes().vec())
    , strides_(tensor.strides().vec()) {}
  CompleteTensorType(at::ScalarType scalar_type, int device, at::IntList sizes)
    : CompleteTensorType(scalar_type, device, sizes, CompleteTensorType::contiguousStridesOf(sizes)) {}
  CompleteTensorType(at::ScalarType scalar_type, int device, at::IntList sizes, at::IntList strides)
    : TensorType(scalar_type, device, sizes.size(), TypeKind::CompleteTensorType)
    , sizes_(sizes.vec())
    , strides_(strides.vec()) {}

  static std::vector<int64_t> contiguousStridesOf(at::IntList sizes) {
    std::vector<int64_t> strides(sizes.size());
    if(sizes.size() == 0) // zero-dim case
      return strides;
    strides.back() = 1;
    for(size_t i = strides.size() - 1; i > 0; i--) {
      strides[i-1] = strides[i] * sizes[i];
    }
    return strides;
  }

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

struct ListType;
using ListTypePtr = std::shared_ptr<ListType>;

struct TORCH_API ListType : public Type {
  // It's not exactly a singleton, but there should be exactly once instance of
  // List[T] for every T
  static constexpr bool is_singleton = true;
  friend struct Type;
  template<typename ... T>
  static ListTypePtr create( T&& ... all ) {
    return ListTypePtr(new ListType( std::forward<T>(all)... ));
  }
  bool operator==(const Type& rhs) const override {
    if(auto rhs_ = rhs.cast<ListType>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }
  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "[]";
    return ss.str();
  }
  TypePtr getElementType() const {
    return elem;
  }
  // common cast List[Tensor]
  static ListTypePtr ofTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofFloats();
private:
  ListType(TypePtr elem)
  : Type(TypeKind::ListType), elem(elem) {}
  static const TypeKind Kind = TypeKind::ListType;
  TypePtr elem;
};

struct TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;

struct TORCH_API TupleType : public Type {
  static constexpr bool is_singleton = false;
  friend struct Type;
  template<typename ... T>
  static TupleTypePtr create( T&& ... all ) {
    return TupleTypePtr(new TupleType( std::forward<T>(all)... ));
  }
  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }
  bool operator==(const Type& rhs) const override {
    return compare(rhs, [](const TypePtr a, const TypePtr b) {
      return *a == *b;
    });
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    // co-variant rules for tuples
    return compare(*rhs, [](const TypePtr a, const TypePtr b) {
      return a->isSubtypeOf(b);
    });
  }
  std::string str() const override {
    std::stringstream ss;
    ss << "(";
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->str();
    }
    ss << ")";
    return ss.str();
  }
private:
  TupleType(std::vector<TypePtr> elements_)
  : Type(TypeKind::TupleType)
  , elements_(std::move(elements_)) {}
  static const TypeKind Kind = TypeKind::TupleType;

  bool compare(const Type& rhs, std::function<bool(const TypePtr, const TypePtr)> fn) const {
    if(rhs.kind() != kind())
      return false;
    const auto & l_elements = elements();
    const auto & r_elements = rhs.cast<TupleType>()->elements();
    if(l_elements.size() != r_elements.size())
      return false;
    for(size_t i = 0; i < l_elements.size(); ++i) {
      if(!fn(l_elements[i], r_elements[i]))
        return false;
    }
    return true;
  }
  std::vector<TypePtr> elements_;
};

struct NumberType;
using NumberTypePtr = std::shared_ptr<NumberType>;
// This node represents a Python number value
struct TORCH_API NumberType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static NumberTypePtr create( T&& ... all ) {
    return NumberTypePtr(new NumberType( std::forward<T>(all)... ));
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Scalar"; // match what PythonArgParser says for clarity
  }
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static NumberTypePtr get();
private:
  NumberType()
  : Type(TypeKind::NumberType) {}
};

struct FloatType;
using FloatTypePtr = std::shared_ptr<FloatType>;
// This node represents a Python float number value
struct TORCH_API FloatType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static FloatTypePtr create( T&& ... all ) {
    return FloatTypePtr(new FloatType( std::forward<T>(all)... ));
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return *this == *rhs || rhs->kind() == TypeKind::NumberType;
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static FloatTypePtr get();
private:
  FloatType()
  : Type(TypeKind::FloatType) {}
};

struct IntType;
using IntTypePtr = std::shared_ptr<IntType>;
// This node represents a Python int number value
struct TORCH_API IntType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static IntTypePtr create( T&& ... all ) {
    return IntTypePtr(new IntType( std::forward<T>(all)... ));
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return *this == *rhs || rhs->kind() == TypeKind::NumberType;
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static IntTypePtr get();
private:
  IntType()
  : Type(TypeKind::IntType) {}
};

struct StringType;
using StringTypePtr = std::shared_ptr<StringType>;
// This node represents a Python string value
struct TORCH_API StringType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static StringTypePtr create( T&& ... all ) {
    return StringTypePtr(new StringType( std::forward<T>(all)... ));
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "string";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return *this == *rhs;
  }
  static const TypeKind Kind = TypeKind::StringType;
  // global singleton
  static StringTypePtr get();
private:
  StringType()
  : Type(TypeKind::StringType) {}
};

struct NoneType;
using NoneTypePtr = std::shared_ptr<NoneType>;
// This node represents a Python int number value
struct NoneType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static NoneTypePtr create( T&& ... all ) {
    return NoneTypePtr(new NoneType( std::forward<T>(all)... ));
  }
  virtual bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  virtual std::string str() const override {
    return "None";
  }
  static const TypeKind Kind = TypeKind::NoneType;
  // global singleton
  static NoneTypePtr get();
private:
  NoneType()
  : Type(TypeKind::NoneType) {}
};

struct GeneratorType;
using GeneratorTypePtr = std::shared_ptr<GeneratorType>;
struct GeneratorType : public Type {
  static constexpr bool is_singleton = true;
  template<typename ... T>
  static GeneratorTypePtr create( T&& ... all) {
    return GeneratorTypePtr(new GeneratorType( std::forward<T>(all)... ));
  }
  virtual bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  virtual std::string str() const override {
    return "Generator";
  }
  static const TypeKind Kind = TypeKind::GeneratorType;
  // global singleton
  static GeneratorTypePtr get();
private:
  GeneratorType()
  : Type(TypeKind::GeneratorType) {}
};


TORCH_API std::ostream& operator<<(std::ostream & out, const Type & t);
// what is the type, ignoring extra size/shape information?
// e.g. Tensor(2x3) -> Dynamic, and Tuple(Tensor(2x3),...) -> Tuple(Dynamic,...)

inline TypePtr unshapedType(const TypePtr& type) {
  if (TupleTypePtr t = type->cast<TupleType>()) {
    return TupleType::create(fmap(t->elements(), unshapedType));
  } else if (ListTypePtr t = type->cast<ListType>()) {
    return ListType::create(unshapedType(t->getElementType()));
  } else if (type->kind() == TypeKind::TensorType ||
             type->kind() == TypeKind::CompleteTensorType) {
    return DynamicType::get();
  } else {
    return type;
  }
}

inline TypePtr CompleteTensorType::fromNumberType(TypePtr typ) {
  JIT_ASSERT(typ->isSubtypeOf(NumberType::get()));
  if (typ->isSubtypeOf(IntType::get())) {
    return CompleteTensorType::create(at::kLong, -1, {});
  } else if (typ->isSubtypeOf(FloatType::get())) {
    return CompleteTensorType::create(at::kFloat, -1, {});
  }
  AT_ERROR("unknown number type", typ->str());
}

// Attempt to find the correct supertype of t1 and t2. If none is found then
// nullopt will be returned. If t1 == t2, or t1 is a type refinement of t2,
// then t2 will be returned (and vice versa).
// Two different tensortypes will return dynamic.
// Currently we chose not to support returning a NumberType for a float & int
// input because of a lack of operator support for NumberType
TORCH_API at::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2);

template <typename T>
TypePtr getTypePtr() {
#define TYPE_STR(Type) #Type, " ",
  AT_ERROR(
      "Type ",
      at::demangle_type<T>(),
      " could not be converted to any of the known types { ",
      TH_FORALL_TYPES(TYPE_STR) "}");
#undef TYPE_STR
  return nullptr;
}

template<> inline TypePtr getTypePtr<at::Tensor>() { return DynamicType::get(); }
template<> inline TypePtr getTypePtr<double>() { return FloatType::get(); }
template<> inline TypePtr getTypePtr<int64_t>() { return IntType::get(); }
template<> inline TypePtr getTypePtr<bool>() { return IntType::get(); }
template<> inline TypePtr getTypePtr<at::Scalar>() { return NumberType::get(); }
template<> inline TypePtr getTypePtr<std::vector<at::Tensor>>() { return ListType::ofTensors(); }
template<> inline TypePtr getTypePtr<std::vector<double>>() { return ListType::ofFloats(); }
template<> inline TypePtr getTypePtr<std::vector<int64_t>>() { return ListType::ofInts(); }

TORCH_API TypePtr inferTypeFrom(const IValue& value);

}} // namespace torch::jit
