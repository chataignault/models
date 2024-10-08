use hdf5::File;
use polars::prelude::*;
use std::path::Path;

fn hdf5_dataset_to_series(dataset: &hdf5::Dataset) -> Result<Series, Box<dyn std::error::Error>> {
    let datatype = dataset.dtype()?;
    let data = match datatype {
        hdf5::Datatype::Float(hdf5::FloatSize::U4) => {
            let data: Vec<f32> = dataset.read_raw()?.as_slice().to_vec();
            Series::new(dataset.name()?, data)
        }
        hdf5::Datatype::Float(hdf5::FloatSize::U8) => {
            let data: Vec<f64> = dataset.read_raw()?.as_slice().to_vec();
            Series::new(dataset.name()?, data)
        }
        hdf5::Datatype::Integer(hdf5::IntSize::U1) => {
            let data: Vec<i8> = dataset.read_raw()?.as_slice().to_vec();
            Series::new(dataset.name()?, data)
        }
        hdf5::Datatype::Integer(hdf5::IntSize::U2) => {
            let data: Vec<i16> = dataset.read_raw()?.as_slice().to_vec();
            Series::new(dataset.name()?, data)
        }
        hdf5::Datatype::Integer(hdf5::IntSize::U4) => {
            let data: Vec<i32> = dataset.read_raw()?.as_slice().to_vec();
            Series::new(dataset.name()?, data)
        }
        hdf5::Datatype::Integer(hdf5::IntSize::U8) => {
            let data: Vec<i64> = dataset.read_raw()?.as_slice().to_vec();
            Series::new(dataset.name()?, data)
        }
        _ => return Err(format!("Unsupported datatype: {:?}", datatype).into()),
    };
    Ok(data)
}

fn hdf5_to_lazy(file_path: &Path, dataset_name: &str) -> Result<LazyFrame, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let group = file.group(dataset_name)?;
    
    let mut series_vec = Vec::new();
    
    for (name, dataset) in group.datasets()? {
        let series = hdf5_dataset_to_series(&dataset)?;
        series_vec.push(series);
    }
    
    let df = DataFrame::new(series_vec)?;
    Ok(df.lazy())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Specify the path to your HDF5 file
    let file_path = Path::new("path/to/your/file.h5");

    // Specify the dataset name within the HDF5 file
    let dataset_name = "your_dataset_name";

    // Open the HDF5 file and convert it to a LazyFrame
    let lf: LazyFrame = hdf5_to_lazy(file_path, dataset_name)?;

    // Print the schema of the LazyFrame
    println!("LazyFrame schema:");
    println!("{}", lf.schema()?);

    // To see the actual data, you can collect and print a few rows
    let df = lf.collect()?;
    println!("\nFirst few rows of the DataFrame:");
    println!("{}", df.head(Some(5)));

    Ok(())
}
