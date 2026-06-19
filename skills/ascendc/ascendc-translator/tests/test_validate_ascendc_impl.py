#!/usr/bin/env python3
"""test_validate_ascendc_impl.py — AST 退化检测 (含 helper 递归检查) UT。

重点验收 (anti-cheat 项目6):
  - 真盲区: forward → self.helper / 模块级 helper 里藏 F.pad/torch.matmul → 仍检出 type3
  - false positive 消除: forward 调本类自定义纯 wrapper 方法 (只调 ext) → 通过
  - 方案A: self.xxx() 非本类方法 (nn.Module 子模块属性) → 仍判违规
  - 递归环不死循环
  - 既有四类退化 (type1-4) 判定保持不变

脚本在 ascendc-translator/scripts 下，测试自插 sys.path 后 import。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from validate_ascendc_impl import validate  # noqa: E402


# 合法基线: 导入 ext + forward 直接调 kernel，无 torch 计算。
_CLEAN = """
import torch
import torch.nn as nn
import my_op_ext

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = torch.empty_like(x)
        my_op_ext.run(x, out)
        return out
"""


def _rtype(code):
    return validate(code)["regression_type"]


def _valid(code):
    return validate(code)["valid"]


class TestBaselineUnchanged(unittest.TestCase):
    """既有判定不回归。"""

    def test_clean_passes(self):
        self.assertTrue(_valid(_CLEAN))

    def test_type1_no_ext_import(self):
        code = """
import torch
import torch.nn as nn
class ModelNew(nn.Module):
    def forward(self, x):
        return x + 1
"""
        self.assertEqual(_rtype(code), 1)

    def test_type2_ext_not_called_in_forward(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        return torch.empty_like(x)
"""
        self.assertEqual(_rtype(code), 2)

    def test_type3_direct_torch_in_forward(self):
        code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        my_op_ext.run(x)
        return F.relu(x)
"""
        self.assertEqual(_rtype(code), 3)

    def test_type4_scalar_for_loop(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        my_op_ext.run(x, out)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                v = x[i, j]
                out[i, j] = v * 2.0 + 1.0 - v / 3.0 + v * v - v
        return out
"""
        self.assertEqual(_rtype(code), 4)


class TestHelperRecursionBlindspot(unittest.TestCase):
    """真盲区: helper 里藏 torch 计算应被递归检出。"""

    def test_self_helper_hides_F_pad(self):
        code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        y = self._compute(x)
        return y
    def _compute(self, x):
        my_op_ext.run(x)
        return F.pad(x, (1, 1))
"""
        # 回归前: forward 只见 self._compute (本类方法不再误判) → 漏检；
        # 递归后: 进入 _compute 检出 F.pad → type3。
        self.assertEqual(_rtype(code), 3)

    def test_module_helper_hides_matmul(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext

def _helper(x):
    return torch.matmul(x, x)

class ModelNew(nn.Module):
    def forward(self, x):
        my_op_ext.run(x)
        return _helper(x)
"""
        self.assertEqual(_rtype(code), 3)

    def test_nested_helper_chain(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        my_op_ext.run(x)
        return self._a(x)
    def _a(self, x):
        return self._b(x)
    def _b(self, x):
        return x.matmul(x)
"""
        self.assertEqual(_rtype(code), 3)


class TestFalsePositiveEliminated(unittest.TestCase):
    """false positive 消除: forward 调本类纯 wrapper 方法应通过。"""

    def test_self_wrapper_only_calls_ext(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        return self._run_kernel(x)
    def _run_kernel(self, x):
        out = torch.empty_like(x)
        my_op_ext.run(x, out)
        return out
"""
        # 回归前: self._run_kernel(...) 被无条件判 type3 (false positive)；
        # 递归后: 是本类方法 → 进入检查，内部只调 ext → 通过。
        self.assertTrue(_valid(code))


class TestSchemeA(unittest.TestCase):
    """方案A: self.xxx() 非本类方法 (nn.Module 子模块) 仍判违规。"""

    def test_nn_submodule_call_is_violation(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        my_op_ext.run(x)
        return self.linear(x)
"""
        self.assertEqual(_rtype(code), 3)


class TestRecursionCycleSafe(unittest.TestCase):
    """递归环不死循环。"""

    def test_mutual_recursion_terminates(self):
        code = """
import torch
import torch.nn as nn
import my_op_ext
class ModelNew(nn.Module):
    def forward(self, x):
        my_op_ext.run(x)
        return self._a(x)
    def _a(self, x):
        return self._b(x)
    def _b(self, x):
        return self._a(x)
"""
        # 不抛 RecursionError；环内无 torch 计算 → valid。
        self.assertTrue(_valid(code))


if __name__ == "__main__":
    unittest.main()
