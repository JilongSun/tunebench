---
description: "Use when coding in this repository through remote SSH sessions with mixed Python paths. Enforces no-environment-operations mode, safe terminal activation checks, and Chinese docs/comments only."
---
# 环境与语言执行规则

## 适用目标

在本仓库进行开发时，优先保证环境稳定，不因代理操作破坏当前解释器与依赖状态。

## 硬性规则

1. 默认只写代码，不操作环境。
- 不主动安装依赖。
- 不主动检查或修改 Python/Conda 环境。
- 不主动执行项目代码、训练命令或评测命令。

2. 若任务明确要求必须打开终端或执行命令，先做环境刷新与解释器确认。
- 先连续执行两次 `conda deactivate`。
- 再执行一次 `conda activate ./.tb311`（在项目根目录下）。
- 再执行 `which python`，确认解释器来自该环境后，才允许继续后续终端操作。

3. 仅文档与注释使用中文。
- 新增或修改的说明文档优先中文。
- 代码中的新增注释优先中文。
- 代码标识符、命令、配置键名、接口字段名按项目技术规范处理，不强制中文。

## 执行偏好

- 若请求可通过静态代码编辑完成，则禁止切换到终端路径。
- 若用户明确要求运行测试或执行脚本，可在完成环境刷新与解释器确认后直接执行。
