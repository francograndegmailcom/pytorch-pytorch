#pragma once
#include <ATen/ATen.h>

// NOTE- if adding new interface functions,
// upodate interpreter.cpp initialize_interface.
static size_t load_model(const char* model_file, bool hermetic=false);
static at::Tensor forward_model(size_t model_id, at::Tensor const & input);
static void run_some_python(const char* code);
static void startup();
static void teardown();
static void run_python_file(const char* code);


#define FOREACH_INTERFACE_FUNCTION(_) \
  _(load_model)                       \
  _(forward_model)                    \
  _(run_some_python)                  \
  _(startup)                          \
  _(teardown)                         \
  _(run_python_file)

struct InterpreterImpl {
#define DEFINE_POINTER(func) decltype(&::func) func;
  FOREACH_INTERFACE_FUNCTION(DEFINE_POINTER)
#undef DEFINE_POINTER
};
