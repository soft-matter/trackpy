#include "Python.h"
#include "numpy/arrayobject.h"

static int
_nullify_secondary_maxima(double *buffer, int filter_size, double *return_value) 
{
  double target_value = *(buffer + filter_size/2);
  if (target_value == 0.0) {
    *return_value = 0.0;
    return 1;
  }
  int i;
  for (i = 1; i < filter_size/2; i++)
  {
    if (*(buffer + i) > target_value)
    {
      *return_value = 0.0;
      return 1;
    }
  }
  for (i = 1 + filter_size/2; i < filter_size; i++)
  {
    if (*(buffer + i) >= target_value)
    {
      *return_value = 0.0;
      return 1;
    }
  }
  *return_value = target_value;
  return 1;
}

static PyObject *
py_nullify_secondary_maxima(PyObject *obj, PyObject *args)
{
  /* wrap function in a CObject: */
  return PyCObject_FromVoidPtr(_nullify_secondary_maxima, NULL);
}

static PyMethodDef methods[] = {
  {"nullify_secondary_maxima", (PyCFunction)py_nullify_secondary_maxima, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_Cfilters(void)
{
  (void) Py_InitModule("_Cfilters", methods);
  import_array();
}

