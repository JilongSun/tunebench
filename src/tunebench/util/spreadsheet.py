"""表格文件读取与 JSON 导出工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SpreadsheetLoadResult:
    """描述一次表格读取与导出的结果。"""

    source_path: Path
    output_path: Path | None
    sheet_name: str | int | None
    records: list[dict[str, Any]]
    row_count: int
    column_names: list[str]


class XlsxJsonConverter:
    """负责读取 xlsx 文件并导出为 JSON 或 JSONL。"""

    def read_records(
        self,
        input_path: str | Path,
        *,
        sheet_name: str | int = 0,
        drop_empty_rows: bool = True,
    ) -> SpreadsheetLoadResult:
        """读取 xlsx 并转换为记录列表。"""
        source_path = Path(input_path)
        dataframe = pd.read_excel(source_path, sheet_name=sheet_name)

        if drop_empty_rows:
            dataframe = dataframe.dropna(how="all")

        # 将缺失值统一替换为 None，便于后续序列化为 JSON。
        normalized = dataframe.where(pd.notna(dataframe), None)
        records = [
            {str(key): value for key, value in record.items()}
            for record in normalized.to_dict(orient="records")
        ]

        return SpreadsheetLoadResult(
            source_path=source_path,
            output_path=None,
            sheet_name=sheet_name,
            records=records,
            row_count=len(records),
            column_names=list(normalized.columns),
        )

    def export_json(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        sheet_name: str | int = 0,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> SpreadsheetLoadResult:
        """将 xlsx 导出为 JSON 数组文件。"""
        result = self.read_records(input_path, sheet_name=sheet_name)
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(result.records, ensure_ascii=ensure_ascii, indent=indent),
            encoding="utf-8",
        )
        result.output_path = target_path
        return result

    def export_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        sheet_name: str | int = 0,
        ensure_ascii: bool = False,
    ) -> SpreadsheetLoadResult:
        """将 xlsx 导出为 JSONL 文件。"""
        result = self.read_records(input_path, sheet_name=sheet_name)
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(record, ensure_ascii=ensure_ascii) for record in result.records]
        target_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        result.output_path = target_path
        return result