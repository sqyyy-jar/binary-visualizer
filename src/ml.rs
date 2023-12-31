use std::{fs::DirEntry, path::Path};

use anyhow::{Error, Result};
use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{loss, ops, Linear, Optimizer, VarBuilder, VarMap};
use log::{info, warn};
use macroquad::rand::ChooseRandom;

use crate::table::BinaryTable;

const N_INPUT: usize = 256 * 256;
const N_HIDDEN_1: usize = 512;
const N_OUTPUT: usize = 5;

const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.01;

#[derive(Clone, Copy, Debug)]
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

    pub fn from_prediction(output: u32) -> Option<Self> {
        match output {
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
    pub fn collect<P>(path: P, dev: &Device) -> Result<Self>
    where
        P: AsRef<Path>,
    {
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
    let ext = path.extension().map(|s| s.to_str().expect("Fuck Windows"));
    let file_type = match ext {
        Some("txt" | "text" | "TXT") => FileType::Text,
        None | Some("bin" | "exe" | "dll" | "so" | "a") => FileType::Binary,
        Some("jpg" | "jpeg") => FileType::Jpeg,
        Some("pdf") => FileType::Pdf,
        Some("wav") => FileType::Wav,
        _ => {
            warn!("Ignoring file with unknown extension {path:?}");
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

    pub fn load<P>(path: P, dev: &Device) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let result = Self::new(vs.clone())?;
        varmap.load(path)?;
        Ok(result)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs).map_err(Into::into)
    }

    pub fn predict(&self, table: &BinaryTable, dev: &Device) -> Result<u32> {
        let input = table.export();
        let input = Tensor::from_vec(input, (1, N_INPUT), dev)?;
        let result = self.forward(&input)?;
        let result = result.argmax(D::Minus1)?.to_dtype(DType::F32)?.get(0)?;
        let output = result.get(0)?.to_dtype(DType::U32)?.to_scalar::<u32>()?;
        Ok(output)
    }
}

pub fn train<P>(m: Dataset, path: P, dev: &Device) -> Result<Network>
where
    P: AsRef<Path>,
{
    let train_inputs = m.train_inputs.to_device(dev)?;
    let train_outputs = m.train_outputs.to_device(dev)?;
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = Network::new(vs.clone())?;
    _ = varmap.load(&path);
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
        info!(
            "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
            loss.to_scalar::<f32>()?,
            final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    _ = varmap.save(&path);
    if final_accuracy < 95.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}
