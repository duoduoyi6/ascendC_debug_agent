
## 📋 Softmax 算子文件生成阶段详细分析

以下是每个文件/目录的生成阶段：

---

### **Stage 1: op-desc-generation** (算子描述生成)

| 文件                     | 说明                    |
| ---------------------- | --------------------- |
| `softmax_op_desc.json` | 算子描述JSON，包含输入/输出/属性定义 |

---

### **Stage 2: reference-generation** (参考代码生成)

| 文件                         | 说明                       |
| -------------------------- | ------------------------ |
| `softmax_reference.py`     | 参考PyTorch实现（nn.Module风格） |


---

### **Stage 3: functional-conversion** (Functional转换)

| 文件 | 说明 |
|------|------|
| `softmax_functional.py` | Functional API风格的PyTorch实现 |

---

### **Stage 4: ascend-call-generation** (Ascend调用生成)

| 文件                                              | 说明                                           |
| ----------------------------------------------- | -------------------------------------------- |
| `softmax_project.json`                          | msopgen项目配置JSON                              |
| `softmax.cpp`                                   | PyBind11绑定代码                                 |
| `softmax_custom.py`                             | ModelNew类，调用 `custom_ops_lib.softmax_custom` |
| `SoftmaxCustom/`                                | **AscendC项目目录** (由 `msopgen gen` 生成)         |
| `SoftmaxCustom/CMakeLists.txt`                  | CMake主配置文件                                   |
| `SoftmaxCustom/CMakePresets.json`               | CMake预设配置                                    |
| `SoftmaxCustom/build.sh`                        | 构建脚本                                         |
| `SoftmaxCustom/cmake/`                          | CMake工具脚本目录                                  |
| `SoftmaxCustom/scripts/`                        | 安装脚本目录                                       |
| `SoftmaxCustom/op_host/`                        | Host侧代码目录                                    |
| `SoftmaxCustom/op_host/CMakeLists.txt`          | Host侧CMake配置                                 |
| `SoftmaxCustom/op_host/softmax_custom.cpp`      | Host侧Tiling实现（初始模板）                          |
| `SoftmaxCustom/op_host/softmax_custom_tiling.h` | Tiling数据结构定义                                 |
| `SoftmaxCustom/op_kernel/`                      | Kernel侧代码目录                                  |
| `SoftmaxCustom/op_kernel/CMakeLists.txt`        | Kernel侧CMake配置                               |
| `SoftmaxCustom/op_kernel/softmax_custom.cpp`    | Kernel实现（初始模板）                               |
| `SoftmaxCustom/framework/`                      | 框架插件目录                                       |
| `SoftmaxCustom/framework/tf_plugin/`            | TensorFlow插件目录                               |

---

### **Stage 5: dsl-baseline-generation** (DSL基线生成)

| 文件 | 说明 |
|------|------|
| `softmax_dsl.py` | AscendDSL Python代码（昇腾DSL风格） |

---

### **Stage 6: dsl-lowering** (DSL降维与编译)

此阶段编译AscendC项目，生成以下内容：

**build_out/ 目录（编译输出）：**

| 文件/目录 | 说明 |
|-----------|------|
| `build_out/CMakeCache.txt` | CMake缓存文件 |
| `build_out/CMakeFiles/` | CMake构建文件 |
| `build_out/CPackConfig.cmake` | CPack打包配置 |
| `build_out/Makefile` | 生成的Makefile |
| `build_out/autogen/` | 自动生成的中间代码 |
| `build_out/op_host/` | Host侧编译输出（.so库） |
| `build_out/op_kernel/` | Kernel侧编译输出（.o二进制） |
| `build_out/framework/` | 框架插件编译输出 |
| `build_out/scripts/` | 安装脚本副本 |
| `build_out/version.info` | 版本信息 |
| `build_out/install_manifest.txt` | 安装清单 |

**安装时生成的 vendors/ 目录：**

| 文件/目录 | 说明 |
|-----------|------|
| `vendors/customize/` | 自定义算子OPP目录（安装.run包时生成） |
| `vendors/customize/bin/set_env.bash` | 环境设置脚本 |
| `vendors/customize/version.info` | 版本信息 |
| `vendors/customize/op_proto/` | 算子原型库 |
| `vendors/customize/op_proto/lib/...` | `libcust_opsproto_rt2.0.so` |
| `vendors/customize/op_proto/inc/op_proto.h` | 原型头文件 |
| `vendors/customize/op_impl/` | 算子实现 |
| `vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/` | Kernel二进制文件(.o) |
| `vendors/customize/op_impl/ai_core/tbe/kernel/config/` | Kernel配置JSON |
| `vendors/customize/op_impl/ai_core/tbe/config/ascend910b/` | ops-info.json |
| `vendors/customize/op_impl/ai_core/tbe/op_tiling/` | Tiling库 |
| `vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/` | TBE动态实现 |
| `vendors/customize/op_api/` | ACLNN API |
| `vendors/customize/op_api/lib/libcust_opapi.so` | API库 |
| `vendors/customize/op_api/include/aclnn_softmax_custom.h` | 头文件 |
| `vendors/customize/framework/tensorflow/` | TF插件 |

---

### **Stage 7: ascendc-evaluation** (PyBind生成与评估)

| 文件/目录 | 说明 |
|-----------|------|
| `ascend_op_pybind/` | PyBind绑定项目 |
| `ascend_op_pybind/CppExtension/setup.py` | Python扩展安装配置 |
| `ascend_op_pybind/CppExtension/csrc/` | C++源码目录 |
| `ascend_op_pybind/CppExtension/csrc/op.cpp` | PyBind绑定代码（从 `softmax.cpp` 复制） |
| `ascend_op_pybind/CppExtension/csrc/pytorch_npu_helper.hpp` | PyTorch NPU辅助头文件 |
| `ascend_op_pybind/CppExtension/build/` | 编译输出目录 |
| `ascend_op_pybind/CppExtension/build/lib.linux-aarch64-cpython-311/` | Python扩展目录 |
| `ascend_op_pybind/CppExtension/build/lib.linux-aarch64-cpython-311/custom_ops_lib.cpython-311-aarch64-linux-gnu.so` | **最终的PyBind库** |
| `ascend_op_pybind/CppExtension/dist/` | Wheel包输出目录 |
| `ascend_op_pybind/CppExtension/dist/custom_ops-1.0-cp311-cp311-linux_aarch64.whl` | Wheel安装包 |
| `evaluation_result.json` | 评估结果文件 |
| `__pycache__/` | Python缓存 |

---

## 📊 生成流程图

```
Stage 1: op_desc_generation
    ↓
    softmax_op_desc.json
    ↓
Stage 2: reference_generation
    ↓
    softmax_reference.py, Softmax_test_cases.jsonl
    ↓
Stage 3: functional_conversion
    ↓
    softmax_functional.py
    ↓
Stage 4: ascend_call_generation
    ↓
    softmax_project.json
    softmax.cpp (PyBind代码)
    softmax_custom.py (ModelNew类)
    SoftmaxCustom/ (AscendC项目骨架 - msopgen生成)
    │   ├── CMakeLists.txt
    │   ├── op_host/ (初始模板)
    │   ├── op_kernel/ (初始模板)
    │   └── ...
    ↓
Stage 5: dsl_baseline_generation
    ↓
    softmax_dsl.py (DSL基线代码)
    ↓
Stage 6: dsl_lowering (4 passes + 编译)
    ↓
    修改 SoftmaxCustom/op_host/*.cpp (tiling_pass)
    修改 SoftmaxCustom/op_kernel/*.cpp (init_pass, process_pass)
    修改 SoftmaxCustom/op_kernel/*.cpp (process_nonaligned_pass)
    ↓
    cmake + make 编译
    ↓
    SoftmaxCustom/build_out/ (编译输出)
    ├── op_host/*.so (Tiling/Proto/API库)
    ├── op_kernel/*.o (Kernel二进制)
    └── framework/*.so (TF插件)
    ↓
    make install / cpack 打包
    ↓
    vendors/packages/ 目录
    custom_opp_ubuntu_aarch64.run (可安装包)
    ↓
    ./custom_opp_ubuntu_aarch64.run --install-path=...
    ↓
    vendors/customize/ (安装后目录)
    ↓
Stage 7: ascendc_evaluation
    ↓
    ascend_op_pybind/ (生成PyBind)
    ├── CppExtension/setup.py
    ├── CppExtension/csrc/ (从softmax.cpp复制)
    └── build/lib.linux-aarch64-cpython-311/custom_ops_lib*.so
    ↓
    pip install custom_ops-*.whl
    ↓
    evaluation_result.json (评估结果)
```

---

## 🎯 关键文件总结

| 文件                       | 生成阶段    | 作用             |
| ------------------------ | ------- | -------------- |
| `*_op_desc.json`         | Stage 1 | 算子元数据定义        |
| `*_reference.py`         | Stage 2 | 参考实现           |
| `*_functional.py`        | Stage 3 | Functional API |
| `*_project.json`         | Stage 4 | AscendC项目配置    |
| `*.cpp` (根目录)            | Stage 4 | PyBind绑定代码     |
| `*_custom.py`            | Stage 4 | Python调用接口     |
| `{Op}Custom/`            | Stage 4 | AscendC项目骨架    |
| `*_dsl.py`               | Stage 5 | DSL基线代码        |
| `build_out/`             | Stage 6 | 编译输出           |
| `vendors/customize/`     | Stage 6 | OPP安装目录        |
| `ascend_op_pybind/`      | Stage 7 | PyBind绑定项目     |
| `evaluation_result.json` | Stage 7 | 评估结果           |
