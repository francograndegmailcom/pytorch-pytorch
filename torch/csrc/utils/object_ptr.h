#pragma once

#include "torch/csrc/utils/python_stub.h"

template<class T>
class THPPointer {
public:
  THPPointer(): ptr(nullptr) {};
  explicit THPPointer(T *ptr): ptr(ptr) {};
  THPPointer(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~THPPointer() { free(); };
  T * get() { return ptr; }
  const T * get() const { return ptr; }
  T * release() { T *tmp = ptr; ptr = nullptr; return tmp; }
  operator T*() { return ptr; }
  THPPointer& operator =(T *new_ptr) { free(); ptr = new_ptr; return *this; }
  THPPointer& operator =(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; return *this; }
  T * operator ->() { return ptr; }
  operator bool() const { return ptr != nullptr; }

private:
  void free();
  T *ptr = nullptr;
};

/**
 * An RAII-style, owning pointer to a PyObject.  You must protect
 * construction and destruction of this object with the GIL.
 *
 * WARNING: Think twice before putting this as a field in a C++
 * struct.  This class does NOT take out the GIL on destruction,
 * so if you will need to ensure that the destructor of your struct
 * is either (a) always invoked when the GIL is taken or (b) takes
 * out the GIL itself.  Easiest way to avoid this problem si to
 * not use THPPointer if you can.
 */
typedef THPPointer<PyObject> THPObjectPtr;
