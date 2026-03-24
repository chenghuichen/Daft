# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations

from typing import TYPE_CHECKING

from daft.api_annotations import PublicAPI
from daft.dataframe import DataFrame

if TYPE_CHECKING:
    from daft.io import IOConfig
    from pypaimon.table.table import Table as PaimonTable


@PublicAPI
def read_paimon(
    table: "PaimonTable",
    io_config: "IOConfig | None" = None,
) -> DataFrame:
    """Create a DataFrame from an Apache Paimon table.

    Args:
        table (pypaimon.table.Table): A Paimon table object created using the pypaimon library.
            Use ``pypaimon.CatalogFactory.create(options).get_table(identifier)`` to obtain one.
        io_config (IOConfig, optional): Unused. Storage credentials are sourced from the
            pypaimon catalog options passed when creating the table.

    Returns:
        DataFrame: a DataFrame with the schema converted from the specified Paimon table.

    Note:
        This function requires the use of `pypaimon <https://pypi.org/project/pypaimon/>`_,
        the Apache Paimon official Python API.

    Examples:
        Read an append-only Paimon table from a local warehouse:

        >>> import pypaimon
        >>> catalog = pypaimon.CatalogFactory.create({"warehouse": "/path/to/warehouse"})
        >>> table = catalog.get_table("mydb.mytable")
        >>> df = daft.read_paimon(table)
        >>> df.show()

        Read a table from an OSS-backed warehouse:

        >>> catalog = pypaimon.CatalogFactory.create(
        ...     {
        ...         "warehouse": "oss://my-bucket/warehouse",
        ...         "fs.oss.endpoint": "oss-cn-hangzhou.aliyuncs.com",
        ...         "fs.oss.accessKeyId": "...",
        ...         "fs.oss.accessKeySecret": "...",
        ...     }
        ... )
        >>> table = catalog.get_table("mydb.mytable")
        >>> df = daft.read_paimon(table)
        >>> df.show()
    """
    try:
        import pypaimon  # noqa: F401
    except ImportError:
        raise ImportError("pypaimon is required to use read_paimon. Install it with: `pip install pypaimon`")

    from daft.io.paimon.paimon_scan import PaimonDataSource

    return PaimonDataSource(table).read()
