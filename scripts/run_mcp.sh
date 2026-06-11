#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/runtime/mcp"
PID_FILE="${RUNTIME_DIR}/tunebench_mcp.pid"
APP_LOG="${PROJECT_ROOT}/runtime/logs/tunebench_mcp.log"
ENV_PATH="${PROJECT_ROOT}/.tb311"

mkdir -p "${RUNTIME_DIR}"

if [[ -f "${PID_FILE}" ]]; then
	existing_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
	if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
		echo "MCP server 已在运行，PID=${existing_pid}"
		exit 0
	fi
	rm -f "${PID_FILE}"
fi

if ! command -v conda >/dev/null 2>&1; then
	echo "未找到 conda 命令，无法启动 MCP server。" >&2
	exit 1
fi

# 直接使用目标环境的 Python 解释器，避免 conda activate 在非交互式 shell 中不可靠
PYTHON_BIN="${ENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
	echo "找不到 Python 解释器: ${PYTHON_BIN}" >&2
	echo "请确认 conda 环境路径是否正确: ${ENV_PATH}" >&2
	exit 1
fi

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TUNEBENCH_MCP_HOST="${TUNEBENCH_MCP_HOST:-127.0.0.1}"
export TUNEBENCH_MCP_PORT="${TUNEBENCH_MCP_PORT:-8888}"
export TUNEBENCH_MCP_PATH="${TUNEBENCH_MCP_PATH:-/mcp}"

# 应用日志由 tunebench.util.logging.setup_logging() 统一写入 runtime/logs/
# MCP 日志: runtime/logs/tunebench_mcp.log
# nohup 的 stderr 重定向仅用于捕获启动早期的致命错误
nohup "${PYTHON_BIN}" -m tunebench_mcp >/dev/null 2>"${RUNTIME_DIR}/startup_errors.log" < /dev/null &
server_pid="$!"
echo "${server_pid}" > "${PID_FILE}"

echo "MCP server 已启动"
echo "PID: ${server_pid}"
echo "APP 日志: ${APP_LOG}"
echo "URL: http://${TUNEBENCH_MCP_HOST}:${TUNEBENCH_MCP_PORT}${TUNEBENCH_MCP_PATH}"