#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/runtime/mcp"
PID_FILE="${RUNTIME_DIR}/tunebench_mcp.pid"
LOG_FILE="${RUNTIME_DIR}/tunebench_mcp.log"
HOST="${TUNEBENCH_MCP_HOST:-127.0.0.1}"
PORT="${TUNEBENCH_MCP_PORT:-8888}"
PATH_SUFFIX="${TUNEBENCH_MCP_PATH:-/mcp}"

if [[ ! -f "${PID_FILE}" ]]; then
	echo "MCP server 未运行。"
	echo "LOG: ${LOG_FILE}"
	echo "URL: http://${HOST}:${PORT}${PATH_SUFFIX}"
	exit 0
fi

server_pid="$(tr -d '[:space:]' < "${PID_FILE}")"

if [[ -z "${server_pid}" ]]; then
	echo "PID 文件为空。"
	echo "LOG: ${LOG_FILE}"
	exit 0
fi

if kill -0 "${server_pid}" 2>/dev/null; then
	echo "MCP server 正在运行。"
	echo "PID: ${server_pid}"
	echo "LOG: ${LOG_FILE}"
	echo "URL: http://${HOST}:${PORT}${PATH_SUFFIX}"
	exit 0
fi

echo "MCP server 未运行，但存在陈旧 PID 文件。"
echo "PID: ${server_pid}"
echo "LOG: ${LOG_FILE}"
echo "URL: http://${HOST}:${PORT}${PATH_SUFFIX}"