# homura-models

Models for [homura](https://crates.io/crates/homura) that use Hugging Face repositories as they
are: a model reads the repository's `config.json` and its weights (`model.safetensors`,
`pytorch_model.bin` or their sharded indexes) without converting them, because its module tree,
its parameter names and its configuration are those of
[transformers](https://github.com/huggingface/transformers).

The first model is transformers' ResNet (`ResNetModel` and `ResNetForImageClassification`).
The logits of `microsoft/resnet-18` and `microsoft/resnet-50` agree with transformers 5.20.2's,
within about 3e-6 in F32 and 5e-15 in F64.

## Loading a model from the Hugging Face cache

homura-models reads repositories offline, from a local directory or from the Hugging Face cache
as `huggingface_hub` downloads it; it never downloads. Download a repository first, with the
`hf` command line tool:

```text
hf download microsoft/resnet-50
```

Then load it:

```rust,no_run
use homura_models::hub::{Repo, Revision};
use homura_models::pretrained::{FromPretrained, PretrainedOptions};
use homura_models::resnet::ResNetForImageClassification;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = Repo::cached("microsoft/resnet-50", Revision::main())?;
    let model = ResNetForImageClassification::from_pretrained(&repo, PretrainedOptions::default())?;
    // Images resized, cropped and normalized as the repository's preprocessor_config.json says.
    let pixel_values = homura::zeros([1, 3, 224, 224]);
    let logits = homura::with_no_grad(|| model.forward(&pixel_values));
    assert_eq!(logits.shape(), [1, 1000]);
    Ok(())
}
```

The cache is found as huggingface_hub finds it (`HF_HUB_CACHE`, `HF_HOME`, then
`~/.cache/huggingface/hub`; `hub::cache_dir` has the details); `Repo::cached_in` takes a cache
directory, and `Repo::local` a directory with the files. `from_pretrained` builds the model on
the meta device and assigns it the weights, so they are read once; `from_pretrained_mmap` maps
them into memory instead (it is `unsafe`: the files must not change while the model uses them).

## One implementation per architecture

Each architecture has one implementation, which follows transformers' model class over its whole
configuration range, so that every repository of that `model_type` loads. Checkpoints of other
libraries with other module trees (torchvision or timm ResNets, say) aren't supported. A model is
built from its configuration at run time, as in transformers, so the layers that a configuration
chooses are enums and `Sequential`s rather than type parameters.

## Differences from transformers

- No downloads, image preprocessing (`preprocessor_config.json`), tokenizers or datasets yet.
- Missing weights are an error: transformers initializes them randomly (a classification head
  loaded from a base model's repository) and warns. There is no `ignore_mismatched_sizes`,
  weight `variant` (`model.fp16.safetensors`), quantization, `device_map` or offloading.
- The ResNet has no `ResNetBackbone`, `output_hidden_states`, `return_dict`, or `labels` and
  loss in `forward`: `ResNetForImageClassification` returns the logits as a tensor, and
  `ResNetModel` the last hidden state and the pooled output.
- Paths are checked more strictly: a file name of a repository can't contain `..`.

## Minimum supported Rust version

Rust 1.88 or later, as homura.

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.
