use std::{
    path::{Path, PathBuf},
    process::exit,
};

use binary_visualizer::{
    ml::{train, Dataset, FileType, Network},
    table::BinaryTable,
};
use candle::Device;
use clap::{arg, command, value_parser};
use log::{error, info, LevelFilter};
use macroquad::{
    prelude::{Color, BLACK},
    shapes::draw_rectangle,
    window::{clear_background, next_frame, Conf},
};

const SCALE: i32 = 4;
const SCALEF: f32 = SCALE as f32;

fn config() -> Conf {
    Conf {
        window_width: 256 * SCALE,
        window_height: 256 * SCALE,
        ..Default::default()
    }
}

fn main() {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .format_timestamp(None)
        .format_target(false)
        .init();
    let matches = command!()
        .subcommands([
            command!("train").alias("t").args([
                arg!(<MODEL> "The file the model is stored in")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
                arg!(<DATA> "The directory of the dataset to train on")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
                arg!(--"accuracy" "The minimum required accuracy expressed in percent (default: 95.0)")
                    .required(false)
                    .value_parser(value_parser!(f32))
                    .default_value("95.0"),
            ]),
            command!("predict").alias("p").args([
                arg!(<MODEL> "The file the model is stored in")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
                arg!(<FILE> "The input file")
                    .required(true)
                    .value_parser(value_parser!(PathBuf))
            ]),
            command!("show")
                .alias("s")
                .args([arg!(<FILE> "The input file")
                    .required(true)
                    .value_parser(value_parser!(PathBuf))]),
        ])
        .subcommand_required(true)
        .get_matches();
    match matches.subcommand() {
        Some(("train", args)) => {
            let model = args.get_one::<PathBuf>("MODEL").unwrap();
            let data = args.get_one::<PathBuf>("DATA").unwrap();
            let &accuracy = args.get_one::<f32>("accuracy").unwrap();
            if !data.exists() || !data.is_dir() {
                error!("The dataset does not exist or is not a directory");
                exit(1);
            }
            if accuracy < 1.0 {
                error!("Minimum accuracy cannot be below 1%");
                exit(1);
            }
            info!("Collecting dataset...");
            let ds = match Dataset::collect(data, &Device::Cpu) {
                Ok(ds) => ds,
                Err(err) => {
                    error!("Could not collect dataset - {err}");
                    exit(1);
                }
            };
            info!("Start training...");
            let _trained_model = loop {
                match train(ds.clone(), model, &Device::Cpu) {
                    Ok(model) => {
                        break model;
                    }
                    Err(err) => {
                        error!("{err}");
                    }
                }
            };
            info!("Model successully trained");
        }
        Some(("predict", args)) => {
            let model = args.get_one::<PathBuf>("MODEL").unwrap();
            let file = args.get_one::<PathBuf>("FILE").unwrap();
            if !model.exists() || !model.is_file() {
                error!("Model does not exist or is not a file");
                exit(1);
            }
            if !file.exists() || !file.is_file() {
                error!("Input does not exist or is not a file");
                exit(1);
            }
            let dev = match Device::cuda_if_available(0) {
                Ok(dev) => dev,
                Err(err) => {
                    error!("Could not create device: {err}");
                    exit(1);
                }
            };
            let model = match Network::load(model, &dev) {
                Ok(model) => model,
                Err(err) => {
                    error!("Could not load model: {err}");
                    exit(1);
                }
            };
            let content = match std::fs::read(file) {
                Ok(content) => content,
                Err(err) => {
                    error!("Could not read input file: {err}");
                    exit(1);
                }
            };
            let mut table = BinaryTable::new();
            table.parse(&content);
            let prediction = match model.predict(&table, &dev) {
                Ok(prediction) => prediction,
                Err(err) => {
                    error!("Could not predict file type: {err}");
                    exit(1);
                }
            };
            let file_type = FileType::from_prediction(prediction);
            info!("{prediction:?} - {file_type:?}");
        }
        Some(("show", args)) => {
            let file = args.get_one::<PathBuf>("FILE").unwrap();
            macroquad::Window::from_config(config(), window(file.clone()));
        }
        _ => unreachable!(),
    }
}

async fn window<P>(path: P)
where
    P: AsRef<Path>,
{
    let bytes = std::fs::read(&path).expect("Read from input file");
    let mut table = BinaryTable::new();
    table.parse(&bytes);
    let export = table.export();
    loop {
        draw(&export);
        next_frame().await
    }
}

fn draw(table: &[f32]) {
    clear_background(BLACK);
    for y in 0..256 {
        for x in 0..256 {
            let t = table[y * 256 + x];
            draw_rectangle(
                x as f32 * SCALEF,
                y as f32 * SCALEF,
                SCALEF,
                SCALEF,
                Color::new(0.0, t, 0.0, 1.0),
            );
        }
    }
}
