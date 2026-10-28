"""Load prepared data; never materialize held-out features or targets in pandas."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Historical Home Credit v4 sizes are absent from its schema.json.
_V4_SPLIT_COUNTS = {"train": 215_257, "validation": 46_127, "test": 46_127}
_STRING_DTYPES = {"str", "string", "object"}


@dataclass
class DataSplit:
    """Features, labels and IDs share the same positional RangeIndex."""

    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series


@dataclass
class PreparedData:
    """Development contains Train followed by Validation, in historical ID order."""

    train: DataSplit
    validation: DataSplit
    development: DataSplit
    feature_names: list[str]
    categorical_features: list[str]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _FeatureSchema:
    """Validated file contract; raw JSON is retained separately for callers."""

    column_names: list[str]
    feature_names: list[str]
    categorical_features: list[str]
    dtypes: dict[str, pa.DataType]
    row_count: int
    split_counts: dict[str, int]
    metadata: dict[str, object]


def _read_schema(path: Path) -> _FeatureSchema:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid schema.json: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError("schema.json must contain an object")

    for key in ("pipeline_version", "id_column", "target_column"):
        if not isinstance(schema.get(key), str) or not schema[key]:
            raise ValueError(f"schema.json: {key} must be a nonempty string")
    if schema["id_column"] != "SK_ID_CURR" or schema["target_column"] != "TARGET":
        raise ValueError("schema.json must use SK_ID_CURR and TARGET")
    for key in ("row_count", "feature_count"):
        if type(schema.get(key)) is not int or schema[key] <= 0:
            raise ValueError(f"schema.json: {key} must be a positive integer")
    for key in ("column_order", "feature_order", "categorical_features"):
        names = schema.get(key)
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"schema.json: {key} must be a list of unique names")

    features = schema["feature_order"]
    columns = schema["column_order"]
    if len(features) != schema["feature_count"]:
        raise ValueError("schema.json: feature_count does not match feature_order")
    if "TARGET" in features or "SK_ID_CURR" in features:
        raise ValueError("schema.json: TARGET and SK_ID_CURR cannot be model features")
    if columns != ["SK_ID_CURR", "TARGET", *features]:
        raise ValueError("schema.json: column_order does not match feature_order")
    dtypes = schema.get("dtypes")
    if not isinstance(dtypes, dict) or set(dtypes) != set(columns):
        raise ValueError("schema.json: dtypes must describe every column exactly")
    parsed_dtypes = {}
    for name, dtype in dtypes.items():
        if not isinstance(dtype, str):
            raise ValueError(f"schema.json: invalid dtype for {name}")
        if dtype in _STRING_DTYPES:
            parsed_dtypes[name] = pa.string()
        else:
            try:
                arrow_type = pa.type_for_alias(dtype)
            except ValueError as exc:
                raise ValueError(f"schema.json: unsupported dtype {dtype!r}") from exc
            if not (
                pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)
            ):
                raise ValueError(f"schema.json: unsupported dtype {dtype!r}")
            parsed_dtypes[name] = arrow_type
    if set(schema["categorical_features"]) != {
        name for name in features if dtypes[name] in _STRING_DTYPES
    }:
        raise ValueError("schema.json: categorical_features must match string features")

    default_counts = _V4_SPLIT_COUNTS if schema["pipeline_version"] == "v4" else None
    counts = schema.get("split_counts", default_counts)
    if (
        not isinstance(counts, dict)
        or set(counts) != set(_V4_SPLIT_COUNTS)
        or any(type(n) is not int or n <= 0 for n in counts.values())
        or sum(counts.values()) != schema["row_count"]
    ):
        raise ValueError("schema.json: invalid split_counts or row_count")
    if schema["pipeline_version"] == "v4" and (
        schema["row_count"] != 307_511
        or len(features) != 138
        or len(schema["categorical_features"]) != 16
        or counts != _V4_SPLIT_COUNTS
    ):
        raise ValueError("schema.json: sizes do not match the fixed v4 dataset")
    return _FeatureSchema(
        column_names=columns,
        feature_names=features,
        categorical_features=schema["categorical_features"],
        dtypes=parsed_dtypes,
        row_count=schema["row_count"],
        split_counts=counts.copy(),
        metadata=schema,
    )


def _validate_feature_schema(actual: pa.Schema, schema: _FeatureSchema) -> None:
    if actual.names != schema.column_names:
        raise ValueError(
            "features.parquet: column names/order do not match schema.json"
        )
    for field in actual:
        expected = schema.dtypes[field.name]
        if pa.types.is_string(expected):
            matches = pa.types.is_string(field.type) or pa.types.is_large_string(
                field.type
            )
        else:
            matches = field.type == expected
        if not matches:
            raise ValueError(
                f"features.parquet: dtype of {field.name} is {field.type}; schema expects {expected}"
            )


def _validate_ids(ids: pd.Series, source: str) -> None:
    if ids.isna().any():
        raise ValueError(f"{source}: SK_ID_CURR contains null IDs")
    if ids.duplicated().any():
        raise ValueError(f"{source}: SK_ID_CURR must be unique; duplicate IDs found")
    if not pd.api.types.is_integer_dtype(ids.dtype) or (ids <= 0).any():
        raise ValueError(f"{source}: SK_ID_CURR must contain positive integer IDs")


def _read_splits(path: Path, schema: _FeatureSchema) -> pd.DataFrame:
    splits = pd.read_parquet(path, engine="pyarrow")
    if set(splits.columns) != {"SK_ID_CURR", "split"}:
        raise ValueError(
            "splits.parquet must contain SK_ID_CURR and split columns only"
        )
    _validate_ids(splits["SK_ID_CURR"], "splits.parquet")
    if (
        splits["split"].isna().any()
        or not splits["split"].isin(schema.split_counts).all()
    ):
        raise ValueError("splits.parquet: split must be train, validation or test")
    actual_counts = splits["split"].value_counts().to_dict()
    if actual_counts != schema.split_counts:
        raise ValueError(
            f"Split counts do not match: expected {schema.split_counts}, got {actual_counts}"
        )
    return splits


def _read_development(
    path: Path, splits: pd.DataFrame, schema: _FeatureSchema
) -> pd.DataFrame:
    """Validate the complete ID universe, then read only development values."""
    _validate_feature_schema(pq.read_schema(path), schema)
    # This projection deliberately reads no Test feature values or targets.
    all_ids = pq.read_table(path, columns=["SK_ID_CURR"]).to_pandas()["SK_ID_CURR"]
    _validate_ids(all_ids, "features.parquet")
    if set(all_ids) != set(splits["SK_ID_CURR"]):
        raise ValueError(
            "SK_ID_CURR mismatch: features.parquet and splits.parquet must have identical IDs"
        )
    if len(all_ids) != schema.row_count:
        raise ValueError("features.parquet: row count does not match schema.json")

    allowed_ids = splits.loc[splits["split"] != "test", "SK_ID_CURR"].tolist()
    frame = pq.read_table(
        path,
        columns=schema.column_names,
        filters=[("SK_ID_CURR", "in", allowed_ids)],
    ).to_pandas()
    for name in schema.categorical_features:
        if frame[name].isna().any():
            raise ValueError(
                f"Categorical feature {name} contains nulls; expected prepared strings"
            )
    return frame.set_index("SK_ID_CURR", drop=False)


def _make_split(
    frame: pd.DataFrame, ids: list[int], features: list[str], name: str
) -> DataSplit:
    ordered = frame.loc[ids].reset_index(drop=True)
    target = ordered["TARGET"]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError(
            f"{name}: TARGET must contain both classes 0 and 1, with no nulls"
        )
    return DataSplit(
        X=ordered[features].copy(), y=target.copy(), ids=ordered["SK_ID_CURR"].copy()
    )


def load_data(data_dir: Path) -> PreparedData:
    """Load a frozen dataset, preserving splits and schema feature order.

    Only SK_ID_CURR is read for all objects. Feature values and TARGET are
    filtered by PyArrow before conversion to pandas, excluding Test rows.
    No feature engineering, split generation, or dtype conversion is applied.

    Raises FileNotFoundError for missing files and ValueError for invalid data.
    v4 sizes are pinned; other versions must declare split_counts in their schema.

    Rows follow the IDs in splits.parquet within each split. Development is
    Train followed by Validation. X, y and ids share a positional RangeIndex.
    metadata contains the original schema.json, without normalization.
    """
    data_dir = Path(data_dir)
    for filename in ("schema.json", "splits.parquet", "features.parquet"):
        if not (data_dir / filename).is_file():
            raise FileNotFoundError(
                f"Required prepared data file not found: {data_dir / filename}"
            )

    schema = _read_schema(data_dir / "schema.json")
    splits = _read_splits(data_dir / "splits.parquet", schema)
    frame = _read_development(data_dir / "features.parquet", splits, schema)

    train_ids = splits.loc[splits["split"] == "train", "SK_ID_CURR"].tolist()
    validation_ids = splits.loc[splits["split"] == "validation", "SK_ID_CURR"].tolist()
    train = _make_split(frame, train_ids, schema.feature_names, "Train")
    validation = _make_split(frame, validation_ids, schema.feature_names, "Validation")
    development = DataSplit(
        X=pd.concat([train.X, validation.X], ignore_index=True),
        y=pd.concat([train.y, validation.y], ignore_index=True),
        ids=pd.concat([train.ids, validation.ids], ignore_index=True),
    )
    return PreparedData(
        train=train,
        validation=validation,
        development=development,
        feature_names=schema.feature_names.copy(),
        categorical_features=schema.categorical_features.copy(),
        metadata=schema.metadata,
    )
