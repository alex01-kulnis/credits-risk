"""Small real Parquet fixtures exercise the loader's public contract."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.data import load_data


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    features = pd.DataFrame(
        {
            "SK_ID_CURR": [90, 11, 32, 22, 12, 91, 31, 21, 14, 34],
            "TARGET": [-1, 0, 1, 1, 1, -1, 0, 0, 0, 1],
            "z_income": [900., 110., 320., 220., 120., 910., 310., 210., 140., 340.],
            "a_job": ["TEST", "a", "b", "b", "b", "TEST", "a", "a", "a", "b"],
        }
    )
    splits = pd.DataFrame(
        {
            "SK_ID_CURR": [34, 12, 90, 21, 31, 11, 91, 32, 14, 22],
            "split": ["validation", "train", "test", "validation", "train",
                      "train", "test", "train", "validation", "validation"],
        }
    )
    schema = {
        "pipeline_version": "fixture",
        "row_count": 10,
        "feature_count": 2,
        "id_column": "SK_ID_CURR",
        "target_column": "TARGET",
        "column_order": ["SK_ID_CURR", "TARGET", "z_income", "a_job"],
        "feature_order": ["z_income", "a_job"],
        "dtypes": {"SK_ID_CURR": "int64", "TARGET": "int64",
                   "z_income": "float64", "a_job": "str"},
        "categorical_features": ["a_job"],
        "split_counts": {"train": 4, "validation": 4, "test": 2},
    }
    features.to_parquet(tmp_path / "features.parquet", index=False)
    splits.to_parquet(tmp_path / "splits.parquet", index=False)
    (tmp_path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return tmp_path


def change_schema(data_dir, change):
    path = data_dir / "schema.json"
    schema = json.loads(path.read_text())
    change(schema)
    path.write_text(json.dumps(schema))


def change_table(data_dir, filename, change):
    path = data_dir / filename
    change(pd.read_parquet(path)).to_parquet(path, index=False)


def test_loads_features_in_schema_order_and_aligns_labels(data_dir):
    result = load_data(data_dir)
    assert result.feature_names == ["z_income", "a_job"]
    assert result.categorical_features == ["a_job"]
    assert result.train.X.columns.tolist() == ["z_income", "a_job"]
    assert result.train.ids.tolist() == [12, 31, 11, 32]
    assert result.train.y.tolist() == [1, 0, 0, 1]
    assert result.train.X.z_income.tolist() == [120., 310., 110., 320.]
    assert result.validation.ids.tolist() == [34, 21, 14, 22]
    assert result.validation.y.tolist() == [1, 0, 0, 1]
    assert result.validation.X.z_income.tolist() == [340., 210., 140., 220.]
    assert result.train.X.index.equals(result.train.y.index)
    assert result.train.X.index.equals(result.train.ids.index)
    assert result.metadata["row_count"] == 10


def test_development_is_train_followed_by_validation_without_test(data_dir):
    result = load_data(data_dir)
    assert result.development.ids.tolist() == [12, 31, 11, 32, 34, 21, 14, 22]
    pd.testing.assert_frame_equal(
        result.development.X,
        pd.concat([result.train.X, result.validation.X], ignore_index=True),
    )
    pd.testing.assert_series_equal(
        result.development.y,
        pd.concat([result.train.y, result.validation.y], ignore_index=True),
    )
    for split in [result.train, result.validation, result.development]:
        assert set(split.ids).isdisjoint({90, 91})
        assert "TEST" not in split.X.a_job.tolist()
        assert set(split.y) == {0, 1}
        assert "TARGET" not in split.X and "SK_ID_CURR" not in split.X


def test_metadata_preserves_original_json_schema(data_dir):
    original_schema = json.loads((data_dir / "schema.json").read_text())
    result = load_data(data_dir)
    assert result.metadata == original_schema


def test_public_feature_name_lists_do_not_alias_metadata(data_dir):
    result = load_data(data_dir)
    result.feature_names.append("new_feature")
    result.categorical_features.clear()
    assert result.metadata["feature_order"] == ["z_income", "a_job"]
    assert result.metadata["categorical_features"] == ["a_job"]
    assert result.train.X.columns.tolist() == ["z_income", "a_job"]


def test_feature_read_filters_out_test_before_pandas_conversion(data_dir, monkeypatch):
    real_read = pq.read_table
    feature_reads = []

    def read_table(*args, **kwargs):
        table = real_read(*args, **kwargs)
        if Path(args[0]).name == "features.parquet" and "TARGET" in table.column_names:
            feature_reads.append(table["SK_ID_CURR"].to_pylist())
        return table

    monkeypatch.setattr(pq, "read_table", read_table)
    load_data(data_dir)
    assert feature_reads
    assert all(set(ids) == {11, 12, 14, 21, 22, 31, 32, 34} for ids in feature_reads)


@pytest.mark.parametrize("filename", ["features.parquet", "splits.parquet", "schema.json"])
def test_missing_file_is_reported(data_dir, filename):
    (data_dir / filename).unlink()
    with pytest.raises(FileNotFoundError, match=filename):
        load_data(data_dir)


@pytest.mark.parametrize("filename", ["features.parquet", "splits.parquet"])
@pytest.mark.parametrize("duplicate_id", [11, 90])
def test_duplicate_ids_including_test_are_rejected(data_dir, filename, duplicate_id):
    change_table(data_dir, filename, lambda df: pd.concat(
        [df, df.loc[df.SK_ID_CURR == duplicate_id]], ignore_index=True))
    with pytest.raises(ValueError, match="unique|duplicate"):
        load_data(data_dir)


@pytest.mark.parametrize("filename", ["features.parquet", "splits.parquet"])
def test_null_ids_are_rejected(data_dir, filename):
    def change(df):
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("Int64")
        df.loc[0, "SK_ID_CURR"] = pd.NA
        return df
    change_table(data_dir, filename, change)
    with pytest.raises(ValueError, match="null|missing"):
        load_data(data_dir)


@pytest.mark.parametrize("filename", ["features.parquet", "splits.parquet"])
def test_missing_or_unexpected_ids_are_rejected(data_dir, filename):
    change_table(data_dir, filename, lambda df: df.replace({"SK_ID_CURR": {90: 999}}))
    with pytest.raises(ValueError, match="ID|SK_ID_CURR"):
        load_data(data_dir)


@pytest.mark.parametrize("label", ["holdout", None])
def test_unknown_or_missing_split_is_rejected(data_dir, label):
    def change(df):
        df.loc[0, "split"] = label
        return df
    change_table(data_dir, "splits.parquet", change)
    with pytest.raises(ValueError, match="split"):
        load_data(data_dir)


def test_incomplete_split_is_rejected(data_dir):
    change_table(data_dir, "splits.parquet", lambda df: df.iloc[:-1])
    with pytest.raises(ValueError, match="ID|rows|count|size"):
        load_data(data_dir)


def test_wrong_split_sizes_are_rejected(data_dir):
    def change(df):
        df.loc[df.SK_ID_CURR == 11, "split"] = "validation"
        return df
    change_table(data_dir, "splits.parquet", change)
    with pytest.raises(ValueError, match="count|size"):
        load_data(data_dir)


def test_id_assigned_to_two_splits_is_rejected(data_dir):
    def change(df):
        row = df.loc[df.SK_ID_CURR == 11].copy()
        row["split"] = "test"
        return pd.concat([df, row], ignore_index=True)
    change_table(data_dir, "splits.parquet", change)
    with pytest.raises(ValueError, match="unique|duplicate|overlap"):
        load_data(data_dir)


@pytest.mark.parametrize("change", [
    lambda s: s.pop("feature_order"),
    lambda s: s.update(feature_count=3),
    lambda s: s.update(feature_order=["a_job", "z_income"]),
    lambda s: s.update(feature_order=["TARGET", "a_job"]),
    lambda s: s.update(feature_order=["SK_ID_CURR", "a_job"]),
    lambda s: s.update(feature_order=["z_income", "z_income"]),
    lambda s: s.update(categorical_features=["missing"]),
    lambda s: s.update(categorical_features=["a_job", "a_job"]),
    lambda s: s.update(categorical_features=[]),
    lambda s: s["dtypes"].update(z_income="int64"),
    lambda s: s["dtypes"].update(z_income="nonsense"),
    lambda s: s.update(row_count=11),
    lambda s: s.update(split_counts={"train": 4, "validation": 4}),
    lambda s: s.update(pipeline_version="v4"),
])
def test_incompatible_schema_is_rejected(data_dir, change):
    change_schema(data_dir, change)
    with pytest.raises(ValueError, match="schema|Schema|v4|dtype|count"):
        load_data(data_dir)


@pytest.mark.parametrize("content", ["{bad", "[]", "null"])
def test_invalid_schema_json_is_rejected(data_dir, content):
    (data_dir / "schema.json").write_text(content)
    with pytest.raises(ValueError, match="schema|Schema"):
        load_data(data_dir)


@pytest.mark.parametrize("column", ["TARGET", "a_job"])
def test_missing_feature_or_target_is_rejected(data_dir, column):
    change_table(data_dir, "features.parquet", lambda df: df.drop(columns=column))
    with pytest.raises(ValueError, match="schema|Schema|TARGET|column"):
        load_data(data_dir)


def test_wrong_physical_column_order_is_rejected(data_dir):
    change_table(data_dir, "features.parquet", lambda df: df[
        ["SK_ID_CURR", "TARGET", "a_job", "z_income"]])
    with pytest.raises(ValueError, match="schema|Schema|order"):
        load_data(data_dir)


@pytest.mark.parametrize("ids", [[11, 12, 31, 32], [14, 21, 22, 34]])
@pytest.mark.parametrize("value", [0, 2, None])
def test_nonbinary_or_single_class_targets_are_rejected(data_dir, ids, value):
    def change(df):
        df["TARGET"] = df["TARGET"].astype("Int64")
        df.loc[df.SK_ID_CURR.isin(ids), "TARGET"] = value
        return df
    change_table(data_dir, "features.parquet", change)
    with pytest.raises(ValueError, match="TARGET|target|classes"):
        load_data(data_dir)
