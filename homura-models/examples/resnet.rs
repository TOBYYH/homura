//! Classifies images with a ResNet of the Hugging Face cache, such as microsoft/resnet-50:
//!
//! ```text
//! cargo run --release -p homura-models --example resnet -- microsoft/resnet-50 input.safetensors
//! ```
//!
//! `input.safetensors` holds `pixel_values`, a `[batch, 3, height, width]` tensor of images that
//! are already preprocessed (resized, cropped and normalized as the repository's
//! `preprocessor_config.json` says; homura doesn't decode or preprocess images). The repository
//! must be in the cache (`hf download microsoft/resnet-50`); the example prints the 5 most likely
//! labels of each image.

use std::error::Error;

use homura::nn::functional::softmax;
use homura::{DType, Tensor};
use homura_models::hub::{Repo, Revision};
use homura_models::pretrained::{FromPretrained, LoadedModel, PretrainedOptions};
use homura_models::resnet::ResNetForImageClassification;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [repo_id, input] = &args[..] else {
        return Err("usage: resnet <repository> <input.safetensors>".into());
    };
    let repo = Repo::cached(repo_id, Revision::main())?;
    let LoadedModel { model, config, .. } =
        ResNetForImageClassification::from_pretrained_with_info(
            &repo,
            PretrainedOptions::default(),
        )?;
    let pixel_values: Tensor = homura::load(input)?
        .get("pixel_values")
        .ok_or("the input has no pixel_values")?
        .clone();
    let dtype = config.common.dtype.unwrap_or(DType::F32);
    let logits = homura::with_no_grad(|| model.forward(&pixel_values.to(dtype)));
    let probabilities = softmax(&logits, 1).to(DType::F64);
    let classes = logits.shape()[1];
    for (image, row) in probabilities.to_vec::<f64>().chunks(classes).enumerate() {
        let mut order: Vec<usize> = (0..classes).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        println!("image {image}:");
        for &class in order.iter().take(5) {
            let label = config
                .common
                .id2label
                .get(&(class as i64))
                .map_or("?", String::as_str);
            println!("  {:6.2}%  {class:4}  {label}", 100.0 * row[class]);
        }
    }
    Ok(())
}
