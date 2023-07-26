# Copyright 2023 The dataclass_array Authors.
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

"""Dataclass Array.

Dataclass arrays are dataclasses which can be sliced, indexed, reshaped,... like
numpy arrays.

"""

import sys

# pylint: disable=g-import-not-at-top,g-bad-import-order,g-importing-member

pytest = sys.modules.get('pytest')
if pytest:
  # Inside tests, rewrite `assert` statement in `dca.testing` for better
  # debug messages
  pytest.register_assert_rewrite('dataclass_array.testing')

from dataclass_array import typing
# TODO(epot): Rename array_field -> field internally
from dataclass_array.array_dataclass import array_field as field
from dataclass_array.array_dataclass import dataclass_array
from dataclass_array.array_dataclass import DataclassArray
from dataclass_array.ops import concat
from dataclass_array.ops import stack
from dataclass_array.vectorization import vectorize_method

# `dca.testing` do not depend on pytest or other heavy deps, so is safe to
# import
from dataclass_array import testing

# A new PyPI release will be pushed everytime `__version__` is increased
# When changing this, also update the CHANGELOG.md
__version__ = '1.5.0'

del sys, pytest
