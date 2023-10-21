use std::env::args;

use macroquad::{
    prelude::{Color, BLACK},
    shapes::draw_rectangle,
    window::{clear_background, next_frame, Conf},
};

pub struct BinaryTable {
    pub max: u32,
    pub dots: Box<[[u32; 256]; 256]>,
}

const IGNORED: [[u8; 2]; 2] = [[0, 0], [255, 255]];
const MAX_AMPLIFICATION: u32 = 1;

impl BinaryTable {
    pub fn parse(bytes: &[u8]) -> Self {
        let mut max = 0;
        let mut dots = Box::new([[0u32; 256]; 256]);
        for window in bytes.windows(2) {
            let xb = window[0];
            let yb = window[1];
            let x = xb as usize;
            let y = yb as usize;
            let value = dots[y][x].saturating_add(1);
            dots[y][x] = value;
            if value > max && !(IGNORED.contains(&[xb, yb])) {
                max = value;
            }
        }
        Self {
            max: (max / MAX_AMPLIFICATION).max(1),
            dots,
        }
    }
}

const SCALE: i32 = 3;
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
    let path = args().nth(1).expect("Input file");
    let bytes = std::fs::read(path).expect("Read from input file");
    let table = BinaryTable::parse(&bytes);
    // draw(&table);
    // let scr = get_screen_data();
    // let texture = Texture2D::from_image(&scr);
    loop {
        // draw_texture_ex(
        //     &texture,
        //     0.0,
        //     0.0,
        //     WHITE,
        //     DrawTextureParams {
        //         flip_y: true,
        //         ..Default::default()
        //     },
        // );
        draw(&table);
        next_frame().await
    }
}

fn draw(table: &BinaryTable) {
    clear_background(BLACK);
    for y in 0..256 {
        for x in 0..256 {
            let value = (table.dots[y][x] as f32 / table.max as f32).min(1.0);
            if value != 0.0 {
                draw_rectangle(
                    x as f32 * SCALEF,
                    y as f32 * SCALEF,
                    SCALEF,
                    SCALEF,
                    Color::new(value, value / 5.0, 0.0, 1.0),
                );
            }
        }
    }
}
