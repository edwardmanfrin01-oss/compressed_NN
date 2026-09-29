use gdcompress::load_csv;
use polars::prelude::*;

#[test]
fn test_load_csv_from_file() {
    let loaded = load_csv("data/tabular/data-10000-8-int.csv", true, None);
    assert!(loaded.is_ok(), "Failed to load CSV file");

    let df = loaded.unwrap();
    assert_eq!(df.height(), 10000, "Expected 10000 rows");
    assert_eq!(df.width(), 8, "Expected 8 columns");
}

#[test]
fn test_load_csv_structure() {
    let df = load_csv("data/tabular/data-10000-8-int.csv", true, None).unwrap();

    assert_eq!(df.width(), 8, "Expected 8 columns");
    assert_eq!(df.height(), 10000, "Expected 10000 rows");

    for col in df.columns() {
        assert_eq!(col.len(), df.height());
    }
}

#[test]
fn test_load_csv_values_are_numeric() {
    let df = load_csv("data/tabular/data-10000-8-int.csv", true, None).unwrap();

    for col in df.columns() {
        let series = col.as_materialized_series();
        let dtype = series.dtype();
        assert!(
            matches!(
                dtype,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Float32
                    | DataType::Float64
            ),
            "Unexpected dtype: {:?}",
            dtype
        );
    }
}

#[test]
fn test_load_csv_values_in_u8_range() {
    let df = load_csv("data/tabular/data-10000-8-int.csv", true, None).unwrap();

    for col in df.columns() {
        let series = col.as_materialized_series();
        let name = series.name().to_string();
        match series.cast(&DataType::Int64) {
            Ok(casted) => {
                let ca = casted.i64().unwrap();
                for v in ca.into_no_null_iter().take(100) {
                    assert!(v <= 255, "column '{}' has value {} > 255", name, v);
                }
            }
            Err(_) => {}
        }
    }
}

#[test]
fn test_load_csv_column_counts_consistent() {
    let df = load_csv("data/tabular/data-10000-8-int.csv", true, None).unwrap();

    let expected_rows = df.height();
    for column in df.columns() {
        assert_eq!(column.len(), expected_rows);
    }
}

#[test]
fn test_load_csv_sample_data_values() {
    let df = load_csv("data/tabular/data-10000-8-int.csv", true, None).unwrap();

    for col in df.columns() {
        let series = col.as_materialized_series();
        let name = series.name().to_string();
        match series.cast(&DataType::Int64) {
            Ok(casted) => {
                let ca = casted.i64().unwrap();
                let first = ca.get(0);
                assert!(first.is_some(), "column '{}' has null at row 0", name);
            }
            Err(_) => {}
        }
    }
}
