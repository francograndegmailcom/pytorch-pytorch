#ifndef THP_COPY_UTILS_H
#define THP_COPY_UTILS_H

#include <functional>
#include <vector>
#include "Types.h"
#include "expand_utils.h"

typedef std::function<void(PyObject*, PyObject*, bool)> THPCopyFunction;
struct THPCopyInfo {
  PyTypeObject* srcType;  // Python type of src tensor/storage
  THPCopyFunction copy;   // copy function
  bool non_blocking;             // true if copy implements an 'non_blocking' copy
  bool broadcast;         // true if the copy implements a broadcast copy
};
typedef std::vector<THPCopyInfo> THPCopyList;

inline bool tryTHPCopy(const THPCopyList& v, PyObject* dst, PyObject* src, bool non_blocking, bool broadcast)
{
  for (auto it = v.begin(); it != v.end(); ++it) {
    if (it->non_blocking == non_blocking && PyType_IsSubtype(Py_TYPE(src), it->srcType)) {
      (it->copy)(dst, src, broadcast);
      return true;
    }
  }
  return false;
}

inline bool THPCopy(const THPCopyList& v, PyObject* dst, PyObject* src, bool non_blocking, bool broadcast)
{
  if (tryTHPCopy(v, dst, src, non_blocking, broadcast)) {
    return true;
  } else if (non_blocking && tryTHPCopy(v, dst, src, false, broadcast)) {
    return true;
  }
  THPUtils_setError("copy from %s to %s isn't implemented",
      THPUtils_typename(src), THPUtils_typename(dst));
  return false;
}

inline PyObject * THPStorageCopyMethod(const THPCopyList& v, PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *src;
  int non_blocking = 0;
  static char *kwlist[] = {"source", "non_blocking", NULL};
  // use int as parse type because bool not available in python2.
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:copy_", kwlist, &src, &non_blocking)) {
    return NULL;
  }

  if (!THPCopy(v, self, src, non_blocking, false)) {
    return NULL;
  }

  Py_INCREF(self);
  return self;
}

inline PyObject * THPTensorCopyMethod(const THPCopyList& v, PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *src;
  int non_blocking = 0;
  int broadcast = 1;
  static char *kwlist[] = {"source", "non_blocking", "broadcast", NULL};
  // use int as parse type because bool not available in python2.
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii:copy_", kwlist, &src, &non_blocking, &broadcast)) {
    return NULL;
  }

  if (!THPCopy(v, self, src, non_blocking, broadcast)) {
    return NULL;
  }

  Py_INCREF(self);
  return self;
}

template <typename StorageDst, typename StorageSrc>
void THPInsertStorageCopyFunction(
  THPCopyList& copyList,
  void (*copyFunc)(LIBRARY_STATE_TYPE StorageDst* x, StorageSrc* z),
  bool non_blocking=false)
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_, bool broadcast) {
    StorageDst* dst = THPTypeInfo<StorageDst>::cdata(dst_);
    StorageSrc* src = THPTypeInfo<StorageSrc>::cdata(src_);

    PyThreadState *_save = NULL;
    try {
      Py_UNBLOCK_THREADS;
      copyFunc(LIBRARY_STATE dst, src);
      Py_BLOCK_THREADS;
    } catch (...) {
      if (_save) {
        Py_BLOCK_THREADS;
      }
      throw;
    }
  };

  PyTypeObject* srcType = THPTypeInfo<StorageSrc>::pyType();
  copyList.push_back({ srcType, wrapper, non_blocking, false });
}

template <typename TensorDst, typename TensorSrc>
void THPInsertTensorCopyFunction(
  THPCopyList& copyList,
  void (*copyFunc)(LIBRARY_STATE_TYPE TensorDst* x, TensorSrc* z),
  bool non_blocking=false,
  bool broadcast=true)
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_, bool broadcast) {
    TensorDst* dst = THPTypeInfo<TensorDst>::cdata(dst_);
    TensorSrc* src = THPTypeInfo<TensorSrc>::cdata(src_);

    TensorSrc *src_save = src;
    THPPointer<TensorSrc> src_guard(newForExpand<TensorSrc>(LIBRARY_STATE_NOARGS));

    // support for "broadcast" parameter to copy_.
    if (broadcast) {
      bool expand_success = false;
      try {
        expand_inplace1<TensorSrc, TensorDst>(LIBRARY_STATE src_guard.get(), src, dst, "src", "dst", true);
        expand_success = true;
      } catch (std::exception &e) {}
      if (expand_success) {
        src = src_guard.get();
      }
    }

    PyThreadState *_save = NULL;
    try {
      Py_UNBLOCK_THREADS;
      copyFunc(LIBRARY_STATE dst, src);
      Py_BLOCK_THREADS;
    } catch (...) {
      if (_save) {
        Py_BLOCK_THREADS;
      }
      throw;
    }
    src = src_save;
  };

  PyTypeObject* srcType = THPTypeInfo<TensorSrc>::pyType();
  copyList.push_back({ srcType, wrapper, non_blocking, broadcast });
}

#endif
