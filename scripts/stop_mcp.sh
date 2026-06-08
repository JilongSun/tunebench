#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/runtime/mcp"
PID_FILE="${RUNTIME_DIR}/tunebench_mcp.pid"

if [[ ! -f "${PID_FILE}" ]]; then
	echo "未找到 MCP server PID 文件，无需停止。"
	exit 0
fi

server_pid="$(tr -d '[:space:]' < "${PID_FILE}")"

if [[ -z "${server_pid}" ]]; then
	rm -f "${PID_FILE}"
	echo "PID 文件为空，已清理。"
	exit 0
fi

if kill -0 "${server_pid}" 2>/dev/null; then
	kill "${server_pid}"
	echo "已发送 MCP server 停止信号，PID=${server_pid}"
else
	echo "MCP server 未运行，已清理陈旧 PID 文件。"
fi

rm -f "${PID_FILE}"