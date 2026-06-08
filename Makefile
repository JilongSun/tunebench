.PHONY: build run stop status

build:
	@echo "创建 conda 环境 .tb311（Python 3.11.15）..."
	@if [ -d ".tb311" ]; then \
		echo "环境 .tb311 已存在，跳过创建。"; \
	else \
		conda create -y -p .tb311 python=3.11.15; \
	fi
	@echo "激活环境并安装 Poetry 依赖..."
	@. .tb311/bin/activate && poetry install

run:
	bash scripts/run_mcp.sh

stop:
	bash scripts/stop_mcp.sh

status:
	bash scripts/status_mcp.sh