import pandas as pd
import pytest

from symphony_dt.schemas.contracts import ColumnRule, TableContract, validate_table


def test_contract_ok() -> None:
    c = TableContract(
        table_name="t",
        primary_key=("id",),
        required=(ColumnRule("id", int, nullable=False), ColumnRule("sex", str, nullable=False, allowed=("M","F"))),
    )
    df = pd.DataFrame({"id":[1], "sex":["M"]})
    out = validate_table(df, c)
    assert out["id"].dtype.name in {"Int64","int64"}


def test_contract_fails_bounds() -> None:
    c = TableContract(table_name="t2", required=(ColumnRule("x", float, min_value=0.0, max_value=1.0),))
    with pytest.raises(ValueError):
        validate_table(pd.DataFrame({"x":[2.0]}), c)
