# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package dyndep
# Module caffe2.python.dyndep
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import os

from caffe2.python import core, extension_loader


def InitOpsLibrary(name):
    """Loads a dynamic library that contains custom operators into Caffe2.

    Since Caffe2 uses static variable registration, you can optionally load a
    separate .so file that contains custom operators and registers that into
    the caffe2 core binary. In C++, this is usually done by either declaring
    dependency during compilation time, or via dynload. This allows us to do
    registration similarly on the Python side.

    Args:
        name: a name that ends in .so, such as "my_custom_op.so". Otherwise,
            the command will simply be ignored.
    Returns:
        None
    """
    if not os.path.exists(name):
        # Note(jiayq): if the name does not exist, instead of immediately
        # failing we will simply print a warning, deferring failure to the
        # time when an actual call is made.
        print('Ignoring {} as it is not a valid file.'.format(name))
        return
    _init_impl(name)


_IMPORTED_DYNDEPS = set()


def GetImportedOpsLibraries():
    return _IMPORTED_DYNDEPS


def _init_impl(path):
    _IMPORTED_DYNDEPS.add(path)
    with extension_loader.DlopenGuard():
        ctypes.CDLL(path)
    # reinitialize available ops
    core.RefreshRegisteredOperators()
