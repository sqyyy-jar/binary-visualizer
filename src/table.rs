pub struct BinaryTable {
    pub max: f32,
    pub dots: Box<[[u32; 256]; 256]>,
}

impl BinaryTable {
    pub fn new() -> Self {
        Self {
            max: 0.0,
            dots: Box::new([[0; 256]; 256]),
        }
    }

    pub fn clear(&mut self) {
        self.max = 0.0;
        for dots in self.dots.iter_mut() {
            dots.fill(0);
        }
    }

    pub fn parse(&mut self, bytes: &[u8]) {
        for window in bytes.windows(2) {
            let xb = window[0];
            let yb = window[1];
            let x = xb as usize;
            let y = yb as usize;
            let value = self.dots[y][x].saturating_add(1);
            self.dots[y][x] = value;
            if value > 0 {
                let f = (value as f32).ln();
                if f > self.max {
                    self.max = f;
                }
            }
        }
    }

    pub fn export(&self) -> Vec<f32> {
        let mut tensor = vec![0f32; 256 * 256];
        for (y, row) in self.dots.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let t = if value == 0 {
                    0.0
                } else {
                    (value as f32).ln() / self.max
                };
                tensor[y * 256 + x] = t;
            }
        }
        tensor
    }
}

impl Default for BinaryTable {
    fn default() -> Self {
        Self::new()
    }
}
