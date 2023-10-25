use std::{ffi::OsStr, fs::DirEntry, os::unix::prelude::OsStrExt};

use anyhow::{Error, Result};
use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{loss, ops, Linear, Optimizer, VarBuilder, VarMap};
use macroquad::rand::ChooseRandom;

use crate::table::BinaryTable;

const N_INPUT: usize = 256 * 256;
const N_HIDDEN_1: usize = 512;
const N_OUTPUT: usize = 5;

const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.005;

#[derive(Clone, Copy)]
pub enum FileType {
    Text,
    Binary,
    Jpeg,
    Pdf,
    Wav,
}

impl FileType {
    pub fn output(self) -> u32 {
        match self {
            Self::Text => 0,
            Self::Binary => 1,
            Self::Jpeg => 2,
            Self::Pdf => 3,
            Self::Wav => 4,
        }
    }

    pub fn from_prediction(output: &[f32]) -> Option<Self> {
        let max = output
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(i, _)| i)?;
        match max {
            0 => Some(Self::Text),
            1 => Some(Self::Binary),
            2 => Some(Self::Jpeg),
            3 => Some(Self::Pdf),
            4 => Some(Self::Wav),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct Dataset {
    pub train_inputs: Tensor,
    pub train_outputs: Tensor,
    pub test_inputs: Tensor,
    pub test_outputs: Tensor,
}

impl Dataset {
    pub fn collect(path: &str, dev: &Device) -> Result<Self> {
        let dir = std::fs::read_dir(path)?;
        let mut files = Vec::new();
        let mut table = BinaryTable::new();
        for sub_dir in dir {
            read_dir(&mut table, &mut files, &sub_dir?)?;
        }
        files.shuffle();
        let len = files.len();
        let train_len = (len as f32 * 0.8) as usize;
        let test_len = len - train_len;
        if train_len == 0 || test_len == 0 {
            return Err(Error::msg("Dataset to small"));
        }
        let mut train_inputs = Vec::new();
        let mut train_outputs = Vec::new();
        let mut test_inputs = Vec::new();
        let mut test_outputs = Vec::new();
        for (i, (typ, input)) in files.into_iter().enumerate() {
            if i < train_len {
                train_inputs.extend(input);
                train_outputs.push(typ.output());
            } else {
                test_inputs.extend(input);
                test_outputs.push(typ.output());
            }
        }
        let train_inputs = Tensor::from_vec(train_inputs, (train_len, N_INPUT), dev)?;
        let train_outputs = Tensor::from_vec(train_outputs, train_len, dev)?;
        let test_inputs = Tensor::from_vec(test_inputs, (test_len, N_INPUT), dev)?;
        let test_outputs = Tensor::from_vec(test_outputs, test_len, dev)?;
        Ok(Self {
            train_inputs,
            train_outputs,
            test_inputs,
            test_outputs,
        })
    }
}

fn read_dir(
    table: &mut BinaryTable,
    files: &mut Vec<(FileType, Vec<f32>)>,
    entry: &DirEntry,
) -> Result<()> {
    let metadata = entry.metadata()?;
    let path = entry.path();
    if metadata.is_dir() {
        let dir = std::fs::read_dir(path)?;
        for sub_dir in dir {
            read_dir(table, files, &sub_dir?)?;
        }
        return Ok(());
    }
    let ext = path.extension().map(OsStr::as_bytes);
    let file_type = match ext {
        Some(b"txt" | b"text" | b"TXT") => FileType::Text,
        None | Some(b"bin" | b"exe" | b"dll" | b"so" | b"a") => FileType::Binary,
        Some(b"jpg" | b"jpeg") => FileType::Jpeg,
        Some(b"pdf") => FileType::Pdf,
        Some(b"wav") => FileType::Wav,
        _ => {
            eprintln!("warning: ignoring file with unknown extension {path:?}");
            return Ok(());
        }
    };
    let bytes = std::fs::read(path)?;
    table.parse(&bytes);
    let input = table.export();
    table.clear();
    files.push((file_type, input));
    Ok(())
}

pub struct Network {
    pub ln1: Linear,
    pub ln2: Linear,
}

impl Network {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(N_INPUT, N_HIDDEN_1, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(N_HIDDEN_1, N_OUTPUT + 1, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs).map_err(Into::into)
    }

    pub fn predict(&self, table: &BinaryTable, dev: &Device) -> Result<[f32; N_OUTPUT]> {
        let input = table.export();
        let input = Tensor::from_vec(input, (1, N_INPUT), dev)?;
        let result = self.forward(&input)?;
        let result = result.argmax(D::Minus1)?.to_dtype(DType::F32)?.get(0)?;
        let mut outputs = [0f32; N_OUTPUT];
        for (i, output) in outputs.iter_mut().enumerate() {
            *output = result.get(i)?.to_scalar::<f32>()?;
        }
        Ok(outputs)
    }
}

pub fn train(m: Dataset, dev: &Device) -> Result<Network> {
    let train_inputs = m.train_inputs.to_device(dev)?;
    let train_outputs = m.train_outputs.to_device(dev)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = Network::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let test_inputs = m.test_inputs.to_device(dev)?;
    let test_outputs = m.test_outputs.to_device(dev)?;
    let mut final_accuracy: f32 = 0.0;
    for epoch in 1..=EPOCHS {
        let logits = model.forward(&train_inputs)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_outputs)?;
        sgd.backward_step(&loss)?;
        let test_logits = model.forward(&test_inputs)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_outputs)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_outputs.dims1()? as f32;
        final_accuracy = 100.0 * test_accuracy;
        eprintln!(
            "info: Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
            loss.to_scalar::<f32>()?,
            final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}
