/* Copyright 2012 Daniel B. Allan
   dallan@pha.jhu.edu, daniel.b.allan@gmail.com
   http://pha.jhu.edu/~dallan
   http://www.danallan.com

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.
 
   You should have received a copy of the GNU General Public License
   along with this program; if not, see <http://www.gnu.org/licenses>. */

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

