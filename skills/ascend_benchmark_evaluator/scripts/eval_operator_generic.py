#!/usr/bin/env python3
"""
通用算子评估脚本 - 支持所有算子类型
支持JSON格式定义的测试用例
"""

import os
import sys
import json
import ast
import torch
import torch_npu
import statistics
import importlib
from typing import List, Dict, Any, Tuple, Callable
from pathlib import Path

# ==================== 配置区域 ====================

# 算子名称 - 通过命令行参数传入
OP_NAME = os.environ.get("OP_NAME", "Add")

# 输出目录路径
OUTPUT_DIR = f"/home/lvyou/y00889327/OpenOps-main/output/{OP_NAME}"

# 数据集目录
DATASET_DIR = "/home/lvyou/y00889327/OpenOps-main/dataset"

# 精度阈值
ATOL = 1e-02
RTOL = 1e-02

# 性能测试配置
WARMUP_ITERATIONS = 10
PERF_ITERATIONS = 50

# ==================== 核心代码 ====================

def set_seed(seed=1024):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


def parse_dtype(dtype_str: str) -> torch.dtype:
    """将字符串dtype转换为torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return dtype_map.get(dtype_str, torch.float32)


def generate_tensor(spec: Dict[str, Any]) -> torch.Tensor:
    """根据spec生成tensor"""
    shape = spec["shape"]
    dtype = parse_dtype(spec["dtype"])
    print(f"生成输入张量: shape={shape}, dtype={dtype}")
    
    # 生成随机数据
    if dtype in [torch.float32, torch.float16, torch.bfloat16]:
        tensor = torch.randn(shape, dtype=dtype)
    elif dtype in [torch.int32, torch.int64]:
        tensor = torch.randint(0, 100, shape, dtype=dtype)
    elif dtype == torch.bool:
        tensor = torch.randint(0, 2, shape, dtype=dtype)
    else:
        tensor = torch.randn(shape, dtype=dtype)
    
    return tensor


def parse_test_case(case: Dict[str, Any]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    解析测试用例，分离tensor输入和属性参数
    
    Returns:
        (tensor_inputs, attrs): tensor输入列表和属性字典
    """
    tensor_inputs = []
    attrs = {}
    
    for input_spec in case["inputs"]:
        if input_spec["type"] == "tensor":
            tensor = generate_tensor(input_spec)
            tensor_inputs.append(tensor)
        elif input_spec["type"] == "attr":
            attrs[input_spec["name"]] = input_spec["value"]
    
    return tensor_inputs, attrs


def extract_init_params(op_name: str, attrs: Dict[str, Any]) -> List[Any]:
    """
    从attrs中提取模型初始化参数
    根据算子类型返回相应的初始化参数列表
    """
    init_params = []
    
    # Softmax: dim是初始化参数
    if "dim" in attrs and op_name.lower() in ["softmax", "logsoftmax"]:
        init_params.append(attrs["dim"])
    
    # LayerNorm: normalized_shape是初始化参数
    if op_name.lower() == "layernorm" and "normalized_shape" in attrs:
        init_params.append(attrs["normalized_shape"])
    
    # AveragePooling2d: kernel_size等是初始化参数
    if op_name.lower() in ["avgpool2d", "averagepooling2d"]:
        if "kernel_size" in attrs:
            init_params.append(attrs["kernel_size"])
        if "stride" in attrs:
            init_params.append(attrs["stride"])
        if "padding" in attrs:
            init_params.append(attrs["padding"])
    
    return init_params


def forward_model(model, inputs: List[torch.Tensor], attrs: Dict[str, Any], op_name: str = ""):
    """
    通用模型前向调用
    根据输入数量和属性自动判断调用方式
    """
    # 获取模型forward方法的参数信息
    import inspect
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    
    # 移除self和fn参数
    if 'self' in params:
        params.remove('self')
    if 'fn' in params:
        params.remove('fn')
    
    # 构建调用参数
    call_args = []
    
    # 处理tensor输入（对应forward的位置参数）
    for i, inp in enumerate(inputs):
        if i < len(params):
            call_args.append(inp)
    
    # 处理属性参数（对应forward的关键字参数）
    call_kwargs = {}
    for key, value in attrs.items():
        if key in params:
            call_kwargs[key] = value
    
    # 调用模型
    return model(*call_args, **call_kwargs)


def evaluate_single_case(
    ref_model: torch.nn.Module,
    custom_model: torch.nn.Module,
    case_idx: int,
    tensor_inputs: List[torch.Tensor],
    attrs: Dict[str, Any],
    device: torch.device,
    op_name: str = ""
) -> Tuple[bool, str]:
    """
    评估单个测试用例
    
    Returns:
        (is_passed, message)
    """
    set_seed(1024)
    
    # 将输入移到NPU
    npu_inputs = [x.to(device) for x in tensor_inputs]
    
    torch_npu.npu.synchronize()
    
    # 执行推理
    with torch.no_grad():
        try:
            ref_output = forward_model(ref_model, npu_inputs, attrs, op_name)
            custom_output = forward_model(custom_model, npu_inputs, attrs, op_name)
        except Exception as e:
            return False, f"Case {case_idx}: 推理失败 - {str(e)}"
    
    torch_npu.npu.synchronize()
    
    # 将结果移回CPU进行比较
    ref_output = ref_output.cpu()
    custom_output = custom_output.cpu()
    
    # 比较结果
    close_mask = torch.isclose(ref_output, custom_output, atol=ATOL, rtol=RTOL)
    matched = close_mask.sum().item()
    total = close_mask.numel()
    match_rate = matched / total * 100
    
    max_diff = (ref_output - custom_output).abs().max().item()
    mean_diff = (ref_output - custom_output).abs().float().mean().item()
    
    shape_str = ",".join(map(str, tensor_inputs[0].shape))
    is_passed = match_rate >= 99.0  # 允许少量误差
    
    message = (
        f"Case {case_idx}: shape=[{shape_str}], dtype={tensor_inputs[0].dtype}, "
        f"attrs={attrs} => "
        f"match_rate={match_rate:.2f}% ({matched}/{total}), "
        f"max_diff={max_diff:.5e}, mean_diff={mean_diff:.5e}"
    )
    
    return is_passed, message


def benchmark_single_case(
    ref_model: torch.nn.Module,
    custom_model: torch.nn.Module,
    tensor_inputs: List[torch.Tensor],
    attrs: Dict[str, Any],
    device: torch.device,
    op_name: str = ""
) -> Tuple[float, float]:
    """
    性能测试单个用例
    
    Returns:
        (ref_time_ms, custom_time_ms)
    """
    set_seed(1024)
    
    npu_inputs = [x.to(device) for x in tensor_inputs]
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        with torch.no_grad():
            _ = forward_model(ref_model, npu_inputs, attrs, op_name)
            _ = forward_model(custom_model, npu_inputs, attrs, op_name)
        torch_npu.npu.synchronize()
    
    # 测试参考实现
    ref_times = []
    for _ in range(PERF_ITERATIONS):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = forward_model(ref_model, npu_inputs, attrs, op_name)
        end.record()
        torch_npu.npu.synchronize()
        ref_times.append(start.elapsed_time(end))
    
    # 测试自定义算子
    custom_times = []
    for _ in range(PERF_ITERATIONS):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = forward_model(custom_model, npu_inputs, attrs, op_name)
        end.record()
        torch_npu.npu.synchronize()
        custom_times.append(start.elapsed_time(end))
    
    ref_median = statistics.median(ref_times)
    custom_median = statistics.median(custom_times)
    
    return ref_median, custom_median


def load_test_cases_from_json(file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载测试用例"""
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    cases = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            try:
                cases.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                continue
    
    return cases


def find_operator_files(dataset_dir: str, op_name: str) -> Tuple[str, str]:
    """
    查找算子文件
    
    支持格式：
    - NPUKernelBench格式: {id}_{OpName}.py 和 {id}_{OpName}.json
    """
    dataset_path = Path(dataset_dir)
    print(f"正在查找算子文件: {op_name} in {dataset_path}")
    
    # 尝试NPUKernelBench格式
    for py_file in dataset_path.glob(f"*_{op_name}.py"):
        json_file = py_file.with_suffix('.json')
        if json_file.exists():
            return str(py_file), str(json_file)
    
    # 尝试小写格式
    for py_file in dataset_path.glob(f"*_{op_name.lower()}.py"):
        json_file = py_file.with_suffix('.json')
        if json_file.exists():
            return str(py_file), str(json_file)
    
    raise FileNotFoundError(f"找不到算子 '{op_name}' 的文件")


def evaluate_all_cases(op_name: str, test_cases: List[Dict], device: torch.device, 
                       ref_model: torch.nn.Module, custom_model: torch.nn.Module):
    """评估所有测试用例"""
    print(f"=" * 80)
    print(f"开始评估算子: {op_name}")
    print(f"测试用例数量: {len(test_cases)}")
    print(f"=" * 80)
    
    passed_count = 0
    failed_count = 0
    case_results = []  # 存储每个测试用例的详细结果
    console_output = []  # 存储打屏输出
    
    for idx, case in enumerate(test_cases):
        tensor_inputs, attrs = parse_test_case(case)
        is_passed, message = evaluate_single_case(
            ref_model, custom_model, idx, tensor_inputs, attrs, device, op_name
        )
        
        status = "✅ PASS" if is_passed else "❌ FAIL"
        output_line = f"{status} {message}"
        print(output_line)
        console_output.append(output_line)
        
        # 保存每个测试用例的详细结果
        case_result = {
            "case_id": idx,
            "shape": list(tensor_inputs[0].shape),
            "dtype": str(tensor_inputs[0].dtype),
            "attrs": attrs,
            "status": "PASS" if is_passed else "FAIL",
            "message": message
        }
        case_results.append(case_result)
        
        if is_passed:
            passed_count += 1
        else:
            failed_count += 1
    
    summary_line1 = f"\n{'=' * 80}"
    summary_line2 = f"正确性评估汇总: 通过 {passed_count}/{len(test_cases)}, 失败 {failed_count}/{len(test_cases)}"
    summary_line3 = f"{'=' * 80}"
    
    print(summary_line1)
    print(summary_line2)
    print(summary_line3)
    
    console_output.extend([summary_line1, summary_line2, summary_line3])
    
    return passed_count == len(test_cases), passed_count, failed_count, case_results, console_output


def benchmark_all_cases(op_name: str, test_cases: List[Dict], device: torch.device,
                        ref_model: torch.nn.Module, custom_model: torch.nn.Module):
    """性能测试所有用例"""
    print(f"\n{'=' * 80}")
    print(f"开始性能测试算子: {op_name}")
    print(f"{'=' * 80}")
    
    print(f"{'Case':<6} {'Shape':<30} {'Ref(ms)':<12} {'Custom(ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for idx, case in enumerate(test_cases[:5]):  # 只测前5个
        tensor_inputs, attrs = parse_test_case(case)
        ref_time, custom_time = benchmark_single_case(
            ref_model, custom_model, tensor_inputs, attrs, device, op_name
        )
        
        speedup = ref_time / custom_time if custom_time > 0 else float('inf')
        shape_str = str(list(tensor_inputs[0].shape))
        
        print(f"{idx:<6} {shape_str:<30} {ref_time:<12.3f} {custom_time:<12.3f} {speedup:<10.2f}x")
    
    print(f"{'=' * 80}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="通用算子评估脚本")
    parser.add_argument("--op", required=True, help="算子名称 (如 Add, Abs, Softmax 等)")
    parser.add_argument("--dataset", default=DATASET_DIR, help=f"数据集目录 (默认: {DATASET_DIR})")
    parser.add_argument("--output", default=None, help="输出目录 (默认: output/{op_name})")
    args = parser.parse_args()
    
    op_name = args.op
    dataset_dir = args.dataset
    output_dir = args.output or f"/home/lvyou/y00889327/OpenOps-main/output/{op_name}"
    
    # 查找测试用例文件
    try:
        ref_file, test_cases_file = find_operator_files(dataset_dir, op_name)
        print(f"找到算子文件: {ref_file}, {test_cases_file}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 添加输出目录到路径
    sys.path.insert(0, output_dir)
    
    # 动态导入模块
    try:
        reference_module = importlib.import_module(f"{op_name}_reference")
        custom_module = importlib.import_module(f"{op_name}_custom")
        
        Model = reference_module.Model
        ModelNew = custom_module.ModelNew
        # 检查自定义算子库是否成功加载
        print(f"\n{'='*80}")
        print("检查自定义算子加载状态...")
        if hasattr(custom_module, '_custom_ops_lib'):
            if custom_module._custom_ops_lib is not None:
                print(f"✅ 自定义算子库已成功加载: {custom_module._custom_ops_lib}")
                print(f"   可用函数: {[attr for attr in dir(custom_module._custom_ops_lib) if not attr.startswith('_')]}")
            else:
                print(f"⚠️  自定义算子库未加载 (值为 None)")
                print(f"   将使用 PyTorch 原生实现作为 fallback")
        else:
            print(f"⚠️  未找到 _custom_ops_lib 属性")
            print(f"   可能正在使用 PyTorch 原生实现")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"导入模块失败: {e}")
        print(f"请确保以下文件存在:")
        print(f"  - {output_dir}/{op_name}_reference.py")
        print(f"  - {output_dir}/{op_name}_custom.py")
        sys.exit(1)
    
    # 加载测试用例
    test_cases = load_test_cases_from_json(test_cases_file)
    print(f"已加载 {len(test_cases)} 个测试用例\n")
    
    # 设置设备
    device = torch.device('npu:0')
    
    # 获取第一个测试用例的attrs，用于提取初始化参数
    _, first_attrs = parse_test_case(test_cases[0])
    init_params = extract_init_params(op_name, first_attrs)
    
    # 创建模型
    try:
        ref_model = Model(*init_params).to(device)
        custom_model = ModelNew(*init_params).to(device)
    except Exception as e:
        print(f"创建模型失败: {e}")
        print(f"尝试无参数初始化...")
        ref_model = Model().to(device)
        custom_model = ModelNew().to(device)
    
    # 运行正确性评估
    all_passed, passed_count, failed_count, case_results, console_output = evaluate_all_cases(
        op_name, test_cases, device, ref_model, custom_model
    )
    
    # 保存结果
    result = {
        "operator": op_name,
        "total_cases": len(test_cases),
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": passed_count / len(test_cases) * 100,
        "all_passed": all_passed,
        "case_results": case_results,
        "console_output": console_output
    }
    
    result_file = Path(output_dir) / "evaluation_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存: {result_file}")
    
    # 如果全部通过，运行性能测试
    if all_passed:
        benchmark_all_cases(op_name, test_cases, device, ref_model, custom_model)
        sys.exit(0)
    else:
        skip_msg = "\n⚠️  存在失败的测试用例，跳过性能测试"
        print(skip_msg)
        # 将跳过的消息也添加到 console_output
        result["console_output"].append(skip_msg)
        # 重新保存包含跳过消息的结果
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
