#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>

#include <utility>
#include <vector>
#include <cstdarg>
#include <exception>
#include <sstream>

#include <torch/csrc/THP.h>

PyObject *THPException_FatalError;

#define ASSERT_TRUE(cond) if (!(cond)) return false
bool THPException_init(PyObject *module)
{
  ASSERT_TRUE(THPException_FatalError = PyErr_NewException("torch.FatalError", nullptr, nullptr));
  ASSERT_TRUE(PyModule_AddObject(module, "FatalError", THPException_FatalError) == 0);
  return true;
}

namespace torch {

void replaceAll(std::string & str,
    const std::string & old_str,
    const std::string & new_str) {
  std::string::size_type pos = 0u;
  while ((pos = str.find(old_str, pos)) != std::string::npos) {
    str.replace(pos, old_str.length(), new_str);
  }
}

std::string processErrorMsg(std::string str) {

  // Translate Aten types to their respective pytorch ones
  std::vector<std::pair<std::string, std::string>> changes {
    {"Variable[SparseCUDAByteType]", "torch.cuda.sparse.ByteTensor"},
    {"Variable[SparseCUDACharType]", "torch.cuda.sparse.CharTensor"},
    {"Variable[SparseCUDADoubleType]", "torch.cuda.sparse.DoubleTensor"},
    {"Variable[SparseCUDAFloatType]", "torch.cuda.sparse.FloatTensor"},
    {"Variable[SparseCUDAIntType]", "torch.cuda.sparse.IntTensor"},
    {"Variable[SparseCUDALongType]", "torch.cuda.sparse.LongTensor"},
    {"Variable[SparseCUDAShortType]", "torch.cuda.sparse.ShortTensor"},
    {"Variable[SparseCUDAHalfType]", "torch.cuda.sparse.HalfTensor"},
    {"Variable[SparseCPUByteType]", "torch.sparse.ByteTensor"},
    {"Variable[SparseCPUCharType]", "torch.sparse.CharTensor"},
    {"Variable[SparseCPUDoubleType]", "torch.sparse.DoubleTensor"},
    {"Variable[SparseCPUFloatType]", "torch.sparse.FloatTensor"},
    {"Variable[SparseCPUIntType]", "torch.sparse.IntTensor"},
    {"Variable[SparseCPULongType]", "torch.sparse.LongTensor"},
    {"Variable[SparseCPUShortType]", "torch.sparse.ShortTensor"},
    {"Variable[SparseCPUHalfType]", "torch.sparse.HalfTensor"},
    {"Variable[CUDAByteType]", "torch.cuda.ByteTensor"},
    {"Variable[CUDACharType]", "torch.cuda.CharTensor"},
    {"Variable[CUDADoubleType]", "torch.cuda.DoubleTensor"},
    {"Variable[CUDAFloatType]", "torch.cuda.FloatTensor"},
    {"Variable[CUDAIntType]", "torch.cuda.IntTensor"},
    {"Variable[CUDALongType]", "torch.cuda.LongTensor"},
    {"Variable[CUDAShortType]", "torch.cuda.ShortTensor"},
    {"Variable[CUDAHalfType]", "torch.cuda.HalfTensor"},
    {"Variable[CPUByteType]", "torch.ByteTensor"},
    {"Variable[CPUCharType]", "torch.CharTensor"},
    {"Variable[CPUDoubleType]", "torch.DoubleTensor"},
    {"Variable[CPUFloatType]", "torch.FloatTensor"},
    {"Variable[CPUIntType]", "torch.IntTensor"},
    {"Variable[CPULongType]", "torch.LongTensor"},
    {"Variable[CPUShortType]", "torch.ShortTensor"},
    {"Variable[CPUHalfType]", "torch.HalfTensor"},
    {"SparseCUDAByteType", "torch.cuda.sparse.ByteTensor"},
    {"SparseCUDACharType", "torch.cuda.sparse.CharTensor"},
    {"SparseCUDADoubleType", "torch.cuda.sparse.DoubleTensor"},
    {"SparseCUDAFloatType", "torch.cuda.sparse.FloatTensor"},
    {"SparseCUDAIntType", "torch.cuda.sparse.IntTensor"},
    {"SparseCUDALongType", "torch.cuda.sparse.LongTensor"},
    {"SparseCUDAShortType", "torch.cuda.sparse.ShortTensor"},
    {"SparseCUDAHalfType", "torch.cuda.sparse.HalfTensor"},
    {"SparseCPUByteType", "torch.sparse.ByteTensor"},
    {"SparseCPUCharType", "torch.sparse.CharTensor"},
    {"SparseCPUDoubleType", "torch.sparse.DoubleTensor"},
    {"SparseCPUFloatType", "torch.sparse.FloatTensor"},
    {"SparseCPUIntType", "torch.sparse.IntTensor"},
    {"SparseCPULongType", "torch.sparse.LongTensor"},
    {"SparseCPUShortType", "torch.sparse.ShortTensor"},
    {"SparseCPUHalfType", "torch.sparse.HalfTensor"},
    {"CUDAByteType", "torch.cuda.ByteTensor"},
    {"CUDACharType", "torch.cuda.CharTensor"},
    {"CUDADoubleType", "torch.cuda.DoubleTensor"},
    {"CUDAFloatType", "torch.cuda.FloatTensor"},
    {"CUDAIntType", "torch.cuda.IntTensor"},
    {"CUDALongType", "torch.cuda.LongTensor"},
    {"CUDAShortType", "torch.cuda.ShortTensor"},
    {"CUDAHalfType", "torch.cuda.HalfTensor"},
    {"CPUByteType", "torch.ByteTensor"},
    {"CPUCharType", "torch.CharTensor"},
    {"CPUDoubleType", "torch.DoubleTensor"},
    {"CPUFloatType", "torch.FloatTensor"},
    {"CPUIntType", "torch.IntTensor"},
    {"CPULongType", "torch.LongTensor"},
    {"CPUShortType", "torch.ShortTensor"},
    {"CPUHalfType", "torch.HalfTensor"},
  };

  for (const auto & it : changes) {
    replaceAll(str, it.first, it.second);
  }

  return str;
}

static std::string formatMessage(const char *format, va_list fmt_args) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);

  // Ensure that the string is null terminated
  error_buf[sizeof(error_buf) / sizeof(*error_buf) - 1] = 0;

  return std::string(error_buf);
}

IndexError::IndexError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

TypeError::TypeError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

ValueError::ValueError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

void PyWarningHandler::process(
    const c10::SourceLocation& source_location,
    const std::string& msg) {
  warning_buffer_.push_back({source_location, msg});
};

PyWarningHandler::PyWarningHandler() noexcept(true):
      prev_handler_(c10::Warning::get_warning_handler()),
      in_exception_(false) {
  c10::Warning::set_warning_handler(this);
}

/// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
/// NOLINTNEXTLINE(bugprone-exception-escape)
PyWarningHandler::~PyWarningHandler() noexcept(false) {
  c10::Warning::set_warning_handler(prev_handler_);

  if(warning_buffer_.size() > 0) {
    if(in_exception_) {
      // An error happened after the warning
      // Simply handle with the previous handler
      for(const auto& warning: warning_buffer_) {
        auto source_location = warning.first;
        const auto& msg = processErrorMsg(warning.second);
        c10::Warning::warn(source_location, msg);
      }
      warning_buffer_.clear();
    } else {
      pybind11::gil_scoped_acquire gil;
      auto result = 0;
      for(const auto& warning: warning_buffer_) {
        auto source_location = warning.first;
        const auto& msg = processErrorMsg(warning.second);
        if (source_location.file == nullptr) {
          result = PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1);
        } else {
          std::string file_name{source_location.file};
          // Sets Python generated warning locations in the attributes
          const std::string python_suffix{".py"};
          if (file_name.size() > python_suffix.size() &&
              std::equal(python_suffix.rbegin(), python_suffix.rend(), file_name.rbegin())) {
            result = PyErr_WarnExplicit(
              /*category=*/PyExc_UserWarning,
              /*message=*/msg.c_str(),
              /*filename=*/source_location.file,
              /*lineno=*/source_location.line,
              /*module=*/nullptr,
              /*registry=*/nullptr);
          } else {
            // Inlines C++ generated warning locations in the message
            std::ostringstream os;
            os << msg << " (Triggered internally at  " << source_location.file;
            os << ":" << source_location.line << ".)";
            result = PyErr_WarnEx(PyExc_UserWarning, os.str().c_str(), 1);
          }
        }
        if (result < 0) {
          break;
        }
      }
      warning_buffer_.clear();
      if (result < 0) {
        /// A warning raised an error, we need to force the parent
        /// function to return an error code.
        throw python_error();
      }
    }
  }
}



} // namespace torch
