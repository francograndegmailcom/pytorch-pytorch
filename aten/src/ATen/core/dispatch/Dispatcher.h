#pragma once

#include <ATen/SequenceNumber.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/LeftRight.h>
#include <list>
#include <mutex>
#include <condition_variable>
#include <type_traits>

#include <ATen/core/grad_mode.h>
#include <ATen/core/enum_tag.h>

namespace c10 {

TORCH_API bool show_dispatch_trace();
TORCH_API void dispatch_trace_nesting_incr();
TORCH_API void dispatch_trace_nesting_decr();
TORCH_API int64_t dispatch_trace_nesting_value();

struct DispatchTraceNestingGuard {
  DispatchTraceNestingGuard() { dispatch_trace_nesting_incr(); }
  ~DispatchTraceNestingGuard() { dispatch_trace_nesting_decr(); }
};

class TORCH_API OperatorHandle;
template<class FuncType> class TypedOperatorHandle;

/**
 * Implement this interface and register your instance with the dispatcher
 * to get notified when operators are registered or deregistered with
 * the dispatcher.
 *
 * NB: registration events only occur when a 'def' occurs; we don't trigger
 * on 'impl' or 'fallback' calls.
 */
class TORCH_API OpRegistrationListener {
public:
  virtual ~OpRegistrationListener();

  virtual void onOperatorRegistered(const OperatorHandle& op) = 0;
  virtual void onOperatorDeregistered(const OperatorHandle& op) = 0;
};

namespace detail {
class RegistrationListenerList;
}
class SchemaRegistrationHandleRAII;

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 * Most end users shouldn't use this directly; if you're trying to register
 * ops look in op_registration
 */
class TORCH_API Dispatcher final {
private:
  // For direct access to backend fallback information
  friend class impl::OperatorEntry;

  struct OperatorDef final {
    explicit OperatorDef(OperatorName&& op_name)
    : op(std::move(op_name)) {}

    impl::OperatorEntry op;

    // These refer to the number of outstanding RegistrationHandleRAII
    // for this operator.  def_count reflects only def() registrations
    // (in the new world, this should only ever be 1, but old style
    // registrations may register the schema multiple times, which
    // will increase this count).  def_and_impl_count reflects the number
    // of combined def() and impl() registrations.  When the last def() gets
    // unregistered, we must immediately call the Deregistered listeners, but we
    // must not actually delete the handle as there are other outstanding RAII
    // destructors which will try to destruct and they had better still have a
    // working operator handle in this case
    size_t def_count = 0;
    size_t def_and_impl_count = 0;
  };
  friend class OperatorHandle;
  template<class> friend class TypedOperatorHandle;

public:
  ~Dispatcher();

  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.
  static Dispatcher& realSingleton();

  C10_ALWAYS_INLINE static Dispatcher& singleton() {
#if !defined C10_MOBILE
    // Implemented inline so that steady-state code needn't incur
    // function-call overhead. We can't just inline `realSingleton`
    // because the function-local static would get duplicated across
    // all DSOs that include & use this header, leading to multiple
    // singleton instances.
    static Dispatcher& s = realSingleton();
    return s;
#else
    // For C10_MOBILE, we should never inline a static function that
    // has a static member, since the generated code calls
    // __cxa_guard_acquire and __cxa_guard_release which help
    // implement exactly once semantics for the initialization of the
    // static Dispatcher& s above (for the non-mobile case). That
    // additional code when duplicated across all operator stubs
    // for every backend results in a lot of additional code
    // being generated by the compiler.
    return realSingleton();
#endif
  }

  // ------------------------------------------------------------------------
  //
  // Accessing operators by schema
  //
  // ------------------------------------------------------------------------

  /**
   * Looks for an operator schema with the given name and overload name
   * and returns it if it is registered WITH A SCHEMA.
   * Returns nullopt otherwise.
   */
  c10::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  /**
   * Variant of findSchema that results in less code generated at the call site.
   * It (1) takes const char* pointer rather than OperatorName (so we skip
   * generating std::string constructor calls at the call site), and (2)
   * it raises an exception if the operator is not found (so we skip
   * generating exception raising code at the call site)
   *
   * Irritatingly, we still have to generate the handful of instructions
   * for dealing with an exception being thrown during static initialization
   * (e.g. __cxa_guard_abort).  If we could annotate this method noexcept we
   * could avoid this code too, but as the name of the function suggests,
   * it does throw exceptions.
   */
  OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);

  // Like findSchema, but also returns OperatorHandle even if there is no schema
  c10::optional<OperatorHandle> findOp(const OperatorName& operator_name);

  // Returns a list of all operator names present in the operatorLookupTable_
  const std::vector<OperatorName> getAllOpNames();

  // ------------------------------------------------------------------------
  //
  // Invoking operators
  //
  // ------------------------------------------------------------------------

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return (Args...)>& op, Args... args) const;


  template<class Return, class... Args>
  static Return callWithDispatchKeySlowPath(const TypedOperatorHandle<Return (Args...)>& op, at::StepCallbacks& stepCallbacks, DispatchKeySet dispatchKeySet, const KernelFunction& kernel, Args... args);

  // Like call, but intended for use in a redispatch in kernels that have explicitly performed the DispatchKey update calculatulation.
  // This will take the DispatchKeySet completely as is and dispatch to the kernel of the corresponding highest priority key in the set.
  // Note that this version of redispatch treats the inputted DispatchKeySet *as is*, and does NOT mask out the highest priority key.
  // See Note [Plumbing Keys Through The Dispatcher]
  template<class Return, class... Args>
  Return redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const;

  // Invoke an operator via the boxed calling convention using an IValue stack
  void callBoxed(const OperatorHandle& op, Stack* stack) const;
  void callBoxedForDispatchKey(const OperatorHandle& op, DispatchKey dk, Stack* stack) const;

  // TODO: This will only be useful if we write a backend fallback that plumbs dispatch keys (currently there are none)
  // See Note [Plumbing Keys Through The Dispatcher]
  void redispatchBoxed(const OperatorHandle& op, DispatchKeySet dispatchKeySet, Stack* stack) const;

  bool hasBackendFallbackForDispatchKey(DispatchKey dk) {
    auto dispatch_ix = getDispatchTableIndexForDispatchKey(dk);
    if (dispatch_ix < 0) return false;
    return backendFallbackKernels_[dispatch_ix].kernel.isValid();
  }

  // Used by torchdeploy/multipy for multiple interpreters racing.
  void waitForDef(const FunctionSchema& schema);
  void waitForImpl(const OperatorName& op_name, c10::optional<DispatchKey> dispatch_key);

  // ------------------------------------------------------------------------
  //
  // Performing registrations (NON user public; use op_registration)
  //
  // ------------------------------------------------------------------------

  /**
   * Register a new operator schema.
   *
   * If a schema with the same operator name and overload name already exists,
   * this function will check that both schemas are exactly identical.
   */
  RegistrationHandleRAII registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags = {});

  /**
   * Register a kernel to the dispatch table for an operator.
   * If dispatch_key is nullopt, then this registers a fallback kernel.
   *
   * @return A RAII object that manages the lifetime of the registration.
   *         Once that object is destructed, the kernel will be deregistered.
   */
  // NB: steals the inferred function schema, as we may need to hold on to
  // it for a bit until the real schema turns up
  RegistrationHandleRAII registerImpl(OperatorName op_name, c10::optional<DispatchKey> dispatch_key, KernelFunction kernel, c10::optional<impl::CppSignature> cpp_signature, std::unique_ptr<FunctionSchema> inferred_function_schema, std::string debug);

  /**
   * Register a new operator by name.
   */
  RegistrationHandleRAII registerName(OperatorName op_name);

  /**
   * Register a fallback kernel for a backend.
   * If an operator is called but there is no concrete kernel for the dispatch
   * key of the given operator arguments, it will check if there is such a
   * fallback kernel for the given dispatch key and, if yes, call that one.
   */
  RegistrationHandleRAII registerFallback(DispatchKey dispatch_key, KernelFunction kernel, std::string debug);

  /**
   * Use to register whenever we had a TORCH_LIBRARY declaration in the frontend
   * API.  These invocations are only permitted once per program, so we raise
   * an error if this is called again for the same namespace.
   */
  RegistrationHandleRAII registerLibrary(std::string ns, std::string debug);

  // ------------------------------------------------------------------------
  //
  // Listeners on registrations
  //
  // ------------------------------------------------------------------------

  /**
   * Add a listener that gets called whenever a new op is registered or an existing
   * op is deregistered. Immediately after registering, this listener gets called
   * for all previously registered ops, so it can be used to keep track of ops
   * registered with this dispatcher.
   */
  RegistrationHandleRAII addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener);

  void checkInvariants() const;

  //
  // ------------------------------------------------------------------------
  //
  // Assertions
  //
  // ------------------------------------------------------------------------

  /**
   * For testing purposes.
   * Returns a list of all operators that were created through calls to registerImpl(),
   * without any corresponding calls to registerDef(). After static initialization
   * is done this is almost certainly a bug, as the created OperatorHandle won't have
   * any schema associated with it and users calling the op through the dispatcher
   * won't be able to access it
   *
   * Note that we cannot enforce this invariant "as we go" during static initialization,
   * due to undefined static initialization order- we have no guarantees over the order
   * in which .def() and .impl() calls are registered in the dispatcher at static
   * initialization time. So this function should only be called after static initialization.
   */
  std::vector<OperatorHandle> findDanglingImpls() const;

  /**
   * Useful for inspecting global Dispatcher registration state.
   * Returns the names of all operators with a kernel registered for the specified DispatchKey.
   * If no DispatchKey is specified, it returns all registered operators.
   */
  std::vector<OperatorName> getRegistrationsForDispatchKey(c10::optional<DispatchKey> k) const;

private:
  Dispatcher();

  static int64_t sequenceNumberForRunningRecordFunction(DispatchKey dispatchKey);
  static void runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey);
  static void runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, c10::ArrayRef<const c10::IValue> args);

  OperatorHandle findOrRegisterSchema_(FunctionSchema&& schema);
  OperatorHandle findOrRegisterName_(const OperatorName& op_name);

  void deregisterDef_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterImpl_(
    const OperatorHandle& op,
    const OperatorName& op_name,
    c10::optional<DispatchKey> dispatch_key,
    impl::OperatorEntry::AnnotatedKernelContainerIterator kernel_handle);
  void deregisterName_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterFallback_(DispatchKey dispatchKey);
  void deregisterLibrary_(const std::string& ns);
  void cleanup(const OperatorHandle& op, const OperatorName& op_name);
  void checkSchemaCompatibility(const OperatorHandle& op, const FunctionSchema& schema, const std::string& debug);

  std::list<OperatorDef> operators_;
#if !defined(C10_MOBILE)
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
#else
  RWSafeLeftRightWrapper<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
#endif
  // Map from namespace to debug string (saying, e.g., where the library was defined)
  ska::flat_hash_map<std::string, std::string> libraries_;

  std::array<impl::AnnotatedKernel, num_runtime_entries> backendFallbackKernels_;

  std::unique_ptr<detail::RegistrationListenerList> listeners_;

  // This mutex protects concurrent access to the dispatcher
  std::mutex mutex_;

  // This condition variable gets notified whenever we add a new def/impl to the
  // dispatch table.  This is primarily used by multipy/torchdeploy, when
  // we have multiple interpreters trying to register to the dispatch table.
  // In this situation, whenever the non-primary interpreter would have tried
  // to register to the dispatch table, instead it will check to see if the
  // expected registration has already been made, and if it hasn't, wait on
  // this condition variable to see if it was just racing with the primary
  // interpreter.
  //
  // We expect it to be rare for there to be any waiters on this condition
  // variable.  This is mostly just to help give better diagnostics if
  // something goes horribly wrong
  std::condition_variable cond_var_;
};

/**
 * This is a handle to an operator schema registered with the dispatcher.
 * This handle can be used to register kernels with the dispatcher or
 * to lookup a kernel for a certain set of arguments.
 */
class TORCH_API OperatorHandle {
  template <typename T> friend struct std::hash;

public:
  OperatorHandle(OperatorHandle&&) noexcept = default;
  OperatorHandle& operator=(OperatorHandle&&) noexcept = default;
  OperatorHandle(const OperatorHandle&) = default;
  OperatorHandle& operator=(const OperatorHandle&) = default;
  // NOLINTNEXTLINE(performance-trivially-destructible)
  ~OperatorHandle();

  const OperatorName& operator_name() const {
    return operatorDef_->op.operator_name();
  }

  bool hasSchema() const {
    return operatorDef_->op.hasSchema();
  }

  const FunctionSchema& schema() const {
    return operatorDef_->op.schema();
  }

  const std::string& debug() const {
    return operatorDef_->op.debug();
  }

  std::string dumpState() const {
    return operatorDef_->op.dumpState();
  }

  bool hasKernelForDispatchKey(DispatchKey k) const {
    return operatorDef_->op.hasKernelForDispatchKey(k);
  }

  bool hasKernelForAnyDispatchKey(DispatchKeySet k) const {
    return operatorDef_->op.hasKernelForAnyDispatchKey(k);
  }

  bool hasComputedKernelForDispatchKey(DispatchKey k) const {
    return operatorDef_->op.hasComputedKernelForDispatchKey(k);
  }

  std::string dumpComputedTable() const {
    return operatorDef_->op.dumpComputedTable();
  }

  void checkInvariants() const {
    return operatorDef_->op.checkInvariants();
  }

  c10::ArrayRef<at::Tag> getTags() const {
    return operatorDef_->op.getTags();
  }

  bool hasTag(const at::Tag& tag) const {
    for(const auto& tag_: getTags()) {
      if (tag == tag_) {
        return true;
      }
    }
    return false;
  }

  template<class FuncType>
  TypedOperatorHandle<FuncType> typed() const {
    // NB: This assert is not 100% sound: you can retrieve a typed() operator
    // handle prior to ANY C++ signature being registered on the operator
    // and the check will say everything is OK (at which point you can then
    // smuggle in a kernel that is typed incorrectly).  For everything
    // in core library this won't happen, because all the static registrations
    // will be done by the time a typed() handle is acquired.
#if !defined C10_MOBILE
    operatorDef_->op.assertSignatureIsCorrect<FuncType>();
#endif
    return TypedOperatorHandle<FuncType>(operatorIterator_);
  }

  void callBoxed(Stack* stack) const {
    c10::Dispatcher::singleton().callBoxed(*this, stack);
  }

  void callBoxed(Stack& stack) const {
    callBoxed(&stack);
  }

  void callBoxedForDispatchKey(DispatchKey dk, Stack& stack) const {
    c10::Dispatcher::singleton().callBoxedForDispatchKey(*this, dk, &stack);
  }

  void redispatchBoxed(DispatchKeySet ks, Stack* stack) const {
    c10::Dispatcher::singleton().redispatchBoxed(*this, ks, stack);
  }

  template <typename F>
  PyObject* getPythonOp(c10::impl::PyInterpreter* self_interpreter, F slow_accessor) const {
    return operatorDef_->op.getPythonOp(self_interpreter, slow_accessor);
  }

  bool operator==(const OperatorHandle& other) const {
    return operatorDef_ == other.operatorDef_;
  }

  bool operator!=(const OperatorHandle& other) const {
    return operatorDef_ != other.operatorDef_;
  }

private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : operatorDef_(&*operatorIterator), operatorIterator_(operatorIterator)  {}
  friend class Dispatcher;
  template<class> friend class TypedOperatorHandle;

  // Storing a direct pointer to the OperatorDef even though we
  // already have the iterator saves an instruction in the critical
  // dispatch path. The iterator is effectively a
  // pointer-to-std::list-node, and (at least in libstdc++'s
  // implementation) the element is at an offset 16 bytes from that,
  // because the prev/next pointers come first in the list node
  // struct. So, an add instruction would be necessary to convert from the
  // iterator to an OperatorDef*.
  Dispatcher::OperatorDef* operatorDef_;

  // We need to store this iterator in order to make
  // Dispatcher::cleanup() fast -- it runs a lot on program
  // termination (and presuambly library unloading).
  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};

/**
 * This is a handle to an operator schema registered with the dispatcher.
 * It holds the same information as an OperatorHandle, but it is templated
 * on the operator arguments and allows calling the operator in an
 * unboxed way.
 */
template<class FuncType>
class TypedOperatorHandle final {
  static_assert(guts::false_t<FuncType>(), "FuncType in OperatorHandle::typed<FuncType> was not a valid function type");
};
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
public:
  TypedOperatorHandle(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle& operator=(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle(const TypedOperatorHandle&) = default;
  TypedOperatorHandle& operator=(const TypedOperatorHandle&) = default;

  // See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
  }

  // See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
  C10_ALWAYS_INLINE Return redispatch(DispatchKeySet currentDispatchKeySet, Args... args) const {
    return c10::Dispatcher::singleton().redispatch<Return, Args...>(*this, currentDispatchKeySet, std::forward<Args>(args)...);
  }

private:
  explicit TypedOperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : OperatorHandle(operatorIterator) {}
  friend class OperatorHandle;
};

namespace detail {
template <class... Args> inline void unused_arg_(const Args&...) {}

// CaptureKernelCall is intended to capture return values from Dispatcher
// unboxed kernel calls. A record function may request to get outputs from the
// kernel calls. For boxed kernels, it's straightforward, the returned values
// are in the stack object. The stack can be passed to record functions. For
// unboxed kernels, we need to handle different kinds of return values, cache
// them temporarily, then release the values for the actual function call
// return.
template <typename ReturnType>
struct CaptureKernelCall {
  template <typename F, typename... Args>
  CaptureKernelCall(
      const F& kernel,
      const TypedOperatorHandle<ReturnType(Args...)>& op,
      const DispatchKeySet& dispatchKeySet,
      Args&&... args)
      // Calls the kernel and capture the result in output_.
      : output_{kernel.template call<ReturnType, Args...>(
            op,
            dispatchKeySet,
            std::forward<Args>(args)...)} {}
  // Wraps the return values in a Stack.
  Stack getOutputs() {
    Stack stack;
    impl::push_outputs<ReturnType, false>::copy(output_, &stack);
    return stack;
  }
  // Since we are returning the output_, we don't expect the output_ to be used
  // afterward. Copy elision and RVO do not apply to class data members. Using
  // move semantic to avoid copies when possible.
  ReturnType release() && {
    return std::move(output_);
  }

 private:
  ReturnType output_;
};

// Handle the lvalue reference differently since it should not be moved.
template <>
inline at::Tensor& CaptureKernelCall<at::Tensor&>::release() && {
  return output_;
}

// Handle case where the kernel returns void.
template <>
struct CaptureKernelCall<void> {
  template <typename F, typename... Args>
  CaptureKernelCall(
      const F& kernel,
      const TypedOperatorHandle<void(Args...)>& op,
      const DispatchKeySet& dispatchKeySet,
      Args&&... args) {
    // Calling the kernel and no need to capture void.
    kernel.template call<void, Args...>(
        op, dispatchKeySet, std::forward<Args>(args)...);
  }
  Stack getOutputs() {
    return Stack();
  }
  void release() && {}
};

} // namespace detail

// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
template<class Return, class... Args>
inline Return Dispatcher::callWithDispatchKeySlowPath(const TypedOperatorHandle<Return(Args...)>& op, at::StepCallbacks& stepCallbacks, DispatchKeySet dispatchKeySet, const KernelFunction& kernel, Args... args) {
  // If callbacks need inputs, we box the arguments and pass them to the guard.
  // Note: For perf reasons we wouldn't want to prematurely box the arguments.
  at::RecordFunction guard(std::move(stepCallbacks));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(op.operatorDef_->op.isObserved());
  auto dispatchKey = dispatchKeySet.highestPriorityTypeId();
  auto& schema = op.schema();
  auto schema_ref = std::reference_wrapper<const FunctionSchema>(schema);
  if (guard.needsInputs()) {
    constexpr auto num_boxed_args = impl::boxed_size<Args...>();
    // If we used std::array<IValue, num_boxed_args> here, we would
    // have to spend time default constructing the IValues in
    // boxedArgs. aligned_storage has no such requirement.
    // Max to avoid zero-size array.`
    std::aligned_storage_t<sizeof(IValue), alignof(IValue)> boxedArgs[std::max(num_boxed_args, static_cast<size_t>(1))];
    // For debugging only; could be removed (but the compiler will do
    // that for us and it's nice to have the extra assurance of
    // correctness from our debug builds).
    int lastArgIdx = 0;
    impl::boxArgsToStack(boxedArgs, lastArgIdx, args...);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(lastArgIdx == num_boxed_args);
    // I don't *think* we need std::launder here, because IValue has
    // no subclasses and no const or reference fields. (We also
    // couldn't use it even if we wanted to because we are currently
    // stuck on C++14 rather than C++17, but we could do a backport
    // similar to folly::launder if needed.)
    runRecordFunction(guard, schema_ref, dispatchKey, c10::ArrayRef<const c10::IValue>(reinterpret_cast<IValue *>(boxedArgs), num_boxed_args));
    for (size_t ii = 0; ii < num_boxed_args; ++ii) {
      reinterpret_cast<IValue *>(&boxedArgs[ii])->~IValue();
    }
  } else {
    runRecordFunction(guard, schema_ref, dispatchKey);
  }

  if (C10_UNLIKELY(guard.needsOutputs())) {
    // Calls the kernel and capture the output temporarily to pass to
    // RecordFunction.
    detail::CaptureKernelCall<Return> captureKernelCall(
        kernel, op, dispatchKeySet, std::forward<Args>(args)...);
    guard.setOutputs(captureKernelCall.getOutputs());
    // Releases the captured output to return to caller.
    return std::move(captureKernelCall).release();
  }

  // keeping the guard alive while executing the kernel
  return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}

// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[call] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && op.operatorDef_->op.isObserved())) {
    return callWithDispatchKeySlowPath<Return, Args...>(op, *step_callbacks, dispatchKeySet, kernel, std::forward<Args>(args)...);
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING
  return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}

// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  // do not use RecordFunction on redispatch
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[redispatch] op=[" << op.operator_name() << "], key=[" << toString(currentDispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  const KernelFunction& kernel = op.operatorDef_->op.lookup(currentDispatchKeySet);
  return kernel.template call<Return, Args...>(op, currentDispatchKeySet, std::forward<Args>(args)...);
}

inline void Dispatcher::callBoxed(const OperatorHandle& op, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const auto& entry = op.operatorDef_->op;
  auto dispatchKeySet = entry.dispatchKeyExtractor().getDispatchKeySetBoxed(stack);
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[callBoxed] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  const auto& kernel = entry.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && entry.isObserved())) {
    at::RecordFunction guard(std::move(*step_callbacks));
    auto dispatchKey = dispatchKeySet.highestPriorityTypeId();
    auto& schema = op.schema();
    auto schema_ref = std::reference_wrapper<const FunctionSchema>(schema);
    guard.needsInputs() ? runRecordFunction(guard, schema_ref, dispatchKey, c10::ArrayRef<const c10::IValue>(stack->data(), stack->size()))
                        : runRecordFunction(guard, schema_ref, dispatchKey);

    // keeping the guard alive while executing the kernel
    kernel.callBoxed(op, dispatchKeySet, stack);

    if (C10_UNLIKELY(guard.needsOutputs())) {
      guard.setOutputs(*stack);
    }
    return;
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING
  kernel.callBoxed(op, dispatchKeySet, stack);
}

// NB: this doesn't count as a "true" dispatcher jump, so no instrumentation
inline void Dispatcher::callBoxedForDispatchKey(const OperatorHandle& op, DispatchKey dk, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const auto& entry = op.operatorDef_->op;
  // We still compute this as we're obligated to pass it on to the internal
  // kernel, if it is a boxed fallback
  auto dispatchKeySet = entry.dispatchKeyExtractor().getDispatchKeySetBoxed(stack);
  const auto& kernel = ([&]() {
    if (op.hasKernelForDispatchKey(dk)) {
      return entry.kernelForDispatchKey(dk);
    } else {
      auto idx = getDispatchTableIndexForDispatchKey(dk);
      TORCH_INTERNAL_ASSERT(idx >= 0);
      return backendFallbackKernels_[idx].kernel;
    }
  })();
  kernel.callBoxed(op, dispatchKeySet, stack);
}

inline void Dispatcher::redispatchBoxed(const OperatorHandle& op, DispatchKeySet dispatchKeySet, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const auto& entry = op.operatorDef_->op;
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[redispatchBoxed] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
  const auto& kernel = entry.lookup(dispatchKeySet);
  return kernel.callBoxed(op, dispatchKeySet, stack);
}

} // namespace c10

namespace std {

template <>
struct hash<c10::OperatorHandle> {
  size_t operator()(c10::OperatorHandle op) const noexcept {
    return std::hash<void*>{}(static_cast<void*>(op.operatorDef_));
  }
};

} // namespace std
