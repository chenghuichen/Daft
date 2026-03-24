from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from daft.dependencies import pa
from daft.io.partitioning import PartitionField
from daft.io.pushdowns import Pushdowns
from daft.io.source import DataSource, DataSourceTask
from daft.logical.schema import Schema
from daft.recordbatch import MicroPartition

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pypaimon.read.split import Split
    from pypaimon.table.file_store_table import FileStoreTable

logger = logging.getLogger(__name__)


class PaimonDataSourceTask(DataSourceTask):
    def __init__(
        self,
        table: FileStoreTable,
        split: Split,
        schema: Schema,
        projected_columns: list[str] | None,
    ) -> None:
        self._table = table
        self._split = split
        self._schema = schema
        self._projected_columns = projected_columns

    @property
    def schema(self) -> Schema:
        return self._schema

    def get_micro_partitions(self) -> Iterator[MicroPartition]:
        read_builder = self._table.new_read_builder()
        if self._projected_columns is not None:
            read_builder = read_builder.with_projection(self._projected_columns)
        table_read = read_builder.new_read()
        reader = table_read.to_arrow_batch_reader([self._split])
        for batch in iter(reader.read_next_batch, None):
            yield MicroPartition.from_arrow_record_batches([batch], reader.schema)


class PaimonDataSource(DataSource):
    def __init__(self, table: FileStoreTable) -> None:
        self._table = table

        from pypaimon.schema.data_types import PyarrowFieldParser

        self._pa_schema: pa.Schema = PyarrowFieldParser.from_paimon_schema(table.fields)
        self._schema = Schema.from_pyarrow_schema(self._pa_schema)

        partition_key_names = set(table.partition_keys)
        self._partition_fields: list[PartitionField] = [
            PartitionField.create(f) for f in self._schema if f.name in partition_key_names
        ]

    @property
    def name(self) -> str:
        return f"PaimonDataSource({getattr(self._table, 'table_path', None)})"

    @property
    def schema(self) -> Schema:
        return self._schema

    def get_partition_fields(self) -> list[PartitionField]:
        return self._partition_fields

    def get_tasks(self, pushdowns: Pushdowns) -> Iterator[PaimonDataSourceTask]:
        read_builder = self._table.new_read_builder()
        projected_columns: list[str] | None = None
        task_schema = self._schema

        if pushdowns.columns is not None:
            partition_keys = self._table.partition_keys
            projected_columns = list(dict.fromkeys(list(pushdowns.columns) + partition_keys))
            read_builder = read_builder.with_projection(projected_columns)
            projected_set = set(projected_columns)
            task_pa_schema = pa.schema([f for f in self._pa_schema if f.name in projected_set])
            task_schema = Schema.from_pyarrow_schema(task_pa_schema)

        if pushdowns.limit is not None:
            read_builder = read_builder.with_limit(pushdowns.limit)

        if self._partition_fields and pushdowns.partition_filters is None:
            logger.warning(
                "%s has partition keys but no partition filter was specified. "
                "This will result in a full table scan.",
                self.name,
            )

        plan = read_builder.new_scan().plan()
        for split in plan.splits():
            if pushdowns.partition_filters is not None and self._partition_fields:
                partition_dict = split.partition.to_dict()
                pv = MicroPartition.from_pydict({k: [v] for k, v in partition_dict.items()})
                if len(pv.filter([pushdowns.partition_filters])) == 0:
                    continue
            yield PaimonDataSourceTask(self._table, split, task_schema, projected_columns)
