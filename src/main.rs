use std::{env::args, process::exit};

use binary_visualizer::{train, BinaryTable, Dataset};
use candle::Device;
use macroquad::{
    prelude::{Color, BLACK},
    shapes::draw_rectangle,
    texture::get_screen_data,
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
    // run().unwrap();
    eprintln!("info: Collecting dataset...");
    let ds = match Dataset::collect("./files", &Device::Cpu) {
        Ok(ds) => ds,
        Err(err) => {
            eprintln!("error: Could not collect dataset - {err}");
            exit(1);
        }
    };
    eprintln!(
        "info: Dataset {{
    train_inputs: {:?}
    train_outputs: {:?}
    test_inputs: {:?}
    test_outputs: {:?}
}}",
        ds.train_inputs.shape(),
        ds.train_outputs.shape(),
        ds.test_inputs.shape(),
        ds.test_outputs.shape(),
    );
    eprintln!("info: Start training...");
    let _trained_model = loop {
        match train(ds.clone(), &Device::Cpu) {
            Ok(model) => {
                break model;
            }
            Err(err) => {
                eprintln!("error: {err}");
            }
        }
    };
    eprintln!("info: Model successully trained");
    if true {
        return;
    }
    macroquad::Window::from_config(config(), window());
}

async fn window() {
    let path = args().nth(1).expect("Input file");
    let bytes = std::fs::read(&path).expect("Read from input file");
    let mut table = BinaryTable::new();
    table.parse(&bytes);
    draw(&table);
    let scr = get_screen_data();
    scr.export_png(&format!("{path}-out.png"));
    loop {
        draw(&table);
        next_frame().await
    }
}

fn draw(table: &BinaryTable) {
    clear_background(BLACK);
    for y in 0..256 {
        for x in 0..256 {
            let t = (table.dots[y][x] as f32).ln() / table.max;
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
