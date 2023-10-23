use std::env::args;

use binary_visualizer::BinaryTable;
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

#[macroquad::main(config)]
async fn main() {
    // run().unwrap();
    // if true {
    //     return;
    // }
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
