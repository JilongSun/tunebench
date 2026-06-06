"""LlamaFactory 分类提示词工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence


_LABEL_DEFINITIONS: dict[str, str] = {
    "ICT巡检": "无具体故障时，对硬件、软件、协议、安全、配置规范、诊断信息、资源容量、集群状态、技术公告、已知问题等做系统性全面检查。",
    "态势查询": "查询运行状态、健康度、指标、趋势预测、TopN、风险预警或特定对象的客观状态点查。",
    "ICT排障": "存在明确故障现象，需要定位原因、分析异常或排查故障。",
    "配置管理": "执行配置写入、修改、删除、下发或模板生成等操作。",
    "知识问答": "获取方法、原理、规格、兼容性、排查思路或命令说明，不直接执行操作。",
    "模糊意图": "用户表达过于笼统，仅有关键词或无法确定查询对象与动作，需要追问澄清。",
    "其它": "与 ICT 领域无关，或不属于上述任一分类。",
}

_GLOBAL_RULES: tuple[str, ...] = (
    "先判断是否存在明确故障现象；有不通、失败、异常、掉线、重启、卡顿、打不开等故障表述时优先考虑 ICT排障。",
    "状态、指标、趋势、TopN、风险预警通常归态势查询；无具体故障的系统性全面检查才归 ICT巡检。",
    "问方法、原理、规格、是否支持、查看命令等通常归知识问答；要求实际修改、下发、设置时归配置管理。",
    "配置规范只读检查归 ICT巡检；配置写操作归配置管理。",
    "仅关键词或表意过于模糊时归模糊意图，并给出 follow_up_question；与 ICT 无关时归其它。",
)

_LABEL_EXAMPLES: dict[str, tuple[str, str, float, str]] = {
    "ICT巡检": (
        "帮我检查所有设备的安全加固情况",
        "用户要求对所有设备的安全加固情况做全量检查，属于无具体故障场景下的系统性巡检。",
        0.9,
        "",
    ),
    "态势查询": (
        "查询一下XX节点光模块的收发光功率",
        "用户查询指定节点光模块的当前指标，属于针对具体状态指标的点查。",
        0.9,
        "",
    ),
    "ICT排障": (
        "MSR5680-X3 ipsec vpn不通",
        "用户描述 IPsec VPN 不通这一明确故障现象，意图是定位原因并排查问题。",
        0.9,
        "",
    ),
    "配置管理": (
        "请帮我在WX2520X上完成配置国家码为中国",
        "用户要求在指定设备上执行国家码配置，属于直接下发或修改配置。",
        0.9,
        "",
    ),
    "知识问答": (
        "MSR3640-X1-HI是否支持NAT64",
        "用户询问设备是否支持 NAT64，属于获取规格与能力信息，不涉及直接操作。",
        0.9,
        "",
    ),
    "模糊意图": (
        "VPN",
        "用户仅给出孤立关键词，缺少明确对象、现象和期望动作，当前无法稳定判定具体一级意图。",
        0.6,
        "请补充是想查询 VPN 状态、排查 VPN 故障、配置 VPN，还是了解 VPN 的相关知识？",
    ),
    "其它": (
        "今天天气怎么样",
        "用户问题与 ICT 设备、网络、配置或运维场景无关。",
        0.9,
        "",
    ),
}


@dataclass(frozen=True, slots=True)
class LabelExample:
    """描述一个标签级 few-shot 示例。"""

    query: str
    label: str
    reasoning: str
    confidence: float
    follow_up_question: str


@dataclass(frozen=True, slots=True)
class LabelGuidance:
    """描述一组标签共享的定义、边界规则与 few-shot。"""

    label_names: tuple[str, ...]
    label_definitions: dict[str, str]
    rules: tuple[str, ...]
    examples: tuple[LabelExample, ...]


def _normalize_label_names(label_names: Sequence[str]) -> tuple[str, ...]:
    normalized_labels: list[str] = []
    seen_labels: set[str] = set()
    for raw_label in label_names:
        normalized_label = str(raw_label).strip()
        if not normalized_label or normalized_label in seen_labels:
            continue
        normalized_labels.append(normalized_label)
        seen_labels.add(normalized_label)
    return tuple(normalized_labels)


def _render_label_definitions(label_names: Sequence[str]) -> str:
    lines: list[str] = []
    for label_name in label_names:
        definition = _LABEL_DEFINITIONS.get(label_name, "请仅在确有充分依据时使用该标签。")
        lines.append(f"- {label_name}: {definition}")
    return "\n".join(lines)


def _render_rules(label_names: Sequence[str]) -> str:
    lines = [f"- {rule}" for rule in _GLOBAL_RULES]
    if "ICT巡检" in label_names and "态势查询" in label_names:
        lines.append("- ICT巡检 与 态势查询 的核心边界：系统性全面检查归 ICT巡检；特定设备、指标、时间段的点查、趋势、TopN、风险预警归态势查询。")
    if "ICT排障" in label_names and "知识问答" in label_names:
        lines.append("- ICT排障 与 知识问答 的核心边界：询问排查方法、可能原因、解决思路归知识问答；已出现明确故障现象并要求定位原因归 ICT排障。")
    if "配置管理" in label_names and "知识问答" in label_names:
        lines.append("- 配置管理 与 知识问答 的核心边界：如何配置、配置样例、命令说明归知识问答；帮我配置、修改、设置、下发归配置管理。")
    if "模糊意图" in label_names:
        lines.append("- 只有在用户信息明显不足、无法稳定分类时才使用模糊意图，并在 follow_up_question 中提出一个最小必要澄清问题。")
    return "\n".join(lines)


def _render_examples(label_names: Sequence[str]) -> str:
    lines: list[str] = []
    for index, example in enumerate(resolve_label_guidance(label_names).examples, start=1):
        label_name = example.label
        follow_up_question = example.follow_up_question
        example_payload = {
            "reasoning": example.reasoning,
            "intents": [
                {
                    "intent": [label_name],
                    "confidence": example.confidence,
                    "follow_up_question": follow_up_question,
                }
            ],
            "intent_relations": None,
        }
        lines.extend(
            [
                f"示例{index}输入: {example.query}",
                f"示例{index}输出: {json.dumps(example_payload, ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def resolve_label_guidance(label_names: Sequence[str]) -> LabelGuidance:
    """按当前标签集合返回统一的分类定义、边界规则与 few-shot。"""
    normalized_label_names = _normalize_label_names(label_names)
    if not normalized_label_names:
        raise ValueError("label_names 不能为空。")

    label_definitions: dict[str, str] = {
        label_name: _LABEL_DEFINITIONS.get(label_name, "请仅在确有充分依据时使用该标签。")
        for label_name in normalized_label_names
    }

    rules: list[str] = list(_GLOBAL_RULES)
    if "ICT巡检" in normalized_label_names and "态势查询" in normalized_label_names:
        rules.append("ICT巡检 与 态势查询 的核心边界：系统性全面检查归 ICT巡检；特定设备、指标、时间段的点查、趋势、TopN、风险预警归态势查询。")
    if "ICT排障" in normalized_label_names and "知识问答" in normalized_label_names:
        rules.append("ICT排障 与 知识问答 的核心边界：询问排查方法、可能原因、解决思路归知识问答；已出现明确故障现象并要求定位原因归 ICT排障。")
    if "配置管理" in normalized_label_names and "知识问答" in normalized_label_names:
        rules.append("配置管理 与 知识问答 的核心边界：如何配置、配置样例、命令说明归知识问答；帮我配置、修改、设置、下发归配置管理。")
    if "模糊意图" in normalized_label_names:
        rules.append("只有在用户信息明显不足、无法稳定分类时才使用模糊意图，并在 follow_up_question 中提出一个最小必要澄清问题。")

    examples: list[LabelExample] = []
    for label_name in normalized_label_names:
        example = _LABEL_EXAMPLES.get(label_name)
        if example is None:
            continue
        query, reasoning, confidence, follow_up_question = example
        examples.append(
            LabelExample(
                query=query,
                label=label_name,
                reasoning=reasoning,
                confidence=confidence,
                follow_up_question=follow_up_question,
            )
        )
    return LabelGuidance(
        label_names=normalized_label_names,
        label_definitions=label_definitions,
        rules=tuple(rules),
        examples=tuple(examples),
    )


def build_instruction(label_names: Sequence[str]) -> str:
    """构建可直接用于结构化分类训练的统一 instruction。"""
    guidance = resolve_label_guidance(label_names)
    normalized_label_names = guidance.label_names

    labels_display = "、".join(normalized_label_names)
    return "\n\n".join(
        [
            "你是 ICT 一级意图分类助手。你的职责是根据用户输入判断最匹配的一级意图，并严格输出结构化 JSON 结果。",
            "任务要求：\n"
            f"1. 只能在以下候选标签中分类：{labels_display}\n"
            "2. 先理解用户真实意图，再依据分类定义与边界做判断。\n"
            "3. reasoning 必须是中文、可展示、120 字以内，不要展开长链路思考。\n"
            "4. 单意图时 intents 只保留一个元素，intent_relations 固定为 null。\n"
            "5. confidence 只能使用 0.90、0.60、0.30 三个值。\n"
            "6. 只有模糊意图允许填写 follow_up_question，其它标签必须为空字符串。",
            "分类定义：\n" + _render_label_definitions(guidance.label_names),
            "关键规则：\n" + _render_rules(guidance.label_names),
            "输出格式：\n"
            "仅输出一个 JSON 对象，不要输出 Markdown、代码块或额外说明。\n"
            "JSON 结构固定为：\n"
            '{"reasoning":"识别逻辑（120字内）","intents":[{"intent":["一级类别"],"confidence":0.90,"follow_up_question":""}],"intent_relations":null}',
            "参考 few-shot：\n" + _render_examples(normalized_label_names),
            "请开始分类，并仅输出 JSON 结果。",
        ]
    )


__all__ = [
    "LabelExample",
    "LabelGuidance",
    "build_instruction",
    "resolve_label_guidance",
]