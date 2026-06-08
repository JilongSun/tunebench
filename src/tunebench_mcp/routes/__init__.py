"""MCP 路由模块。

按功能域划分路由：
- execute: 阶段操作执行（数据准备、推理增强、结构化目标、训练、评测）
- manage: Workflow 生命周期管理（创建、预览、状态查询、日志）
- monitor: 系统监控工具（如显卡使用查询）
- assets: MCP Resources（asset 数据暴露）
"""

from . import assets, execute, manage, utility

__all__ = ["execute", "manage", "utility", "assets"]
