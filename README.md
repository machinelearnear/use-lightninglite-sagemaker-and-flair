# Distributed training with LightningLite, SageMaker, and Flair

## What's the problem?

Currently, [flair](https://github.com/flairNLP/flair/tree/master/flair) does not support multi-gpu model training with its two `Trainers` (a) `language_model_trainer.py` and (b) `trainer.py` (NER, POS, Classification, etc.). There is a PR [under review](https://github.com/flairNLP/flair/pull/2700) to get (a) to work with Lightning Lite as a stepping stone towards full compatibility. This repo tries to get (b) to work.

## What's PyTorch Lightning Lite?
From their [website](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html):

> `LightningLite` enables pure PyTorch users to scale their existing code on any kind of device while retaining full control over their own loops and optimization logic.

> `LightningLite` is the right tool for you if you match one of the two following descriptions:
> - I want to quickly scale my existing code to multiple devices with minimal code changes.
> - I would like to convert my existing code to the Lightning API, but a full path to Lightning transition might be too complex. 
> - I am looking for a stepping stone to ensure reproducibility during the transition.

### Convert to LightningLite
Here are five required steps to convert to LightningLite.

- Subclass LightningLite and override its run() method.
- Move the body of your existing run function into LightningLite run method.
- Remove all .to(...), .cuda() etc calls since LightningLite will take care of it.
- Apply setup() over each model and optimizers pair and setup_dataloaders() on all your dataloaders and replace loss.backward() by self.backward(loss).
- Instantiate your LightningLite subclass and call its run() method.

This is an example
```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.lite import LightningLite


class MyModel(nn.Module):
    ...


class MyDataset(Dataset):
    ...


class Lite(LightningLite):
    def run(self, args):

        model = MyModel(...)
        optimizer = torch.optim.SGD(model.parameters(), ...)
        model, optimizer = self.setup(model, optimizer)  # Scale your model / optimizers

        dataloader = DataLoader(MyDataset(...), ...)
        dataloader = self.setup_dataloaders(dataloader)  # Scale your dataloaders

        model.train()
        for epoch in range(args.num_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                self.backward(loss)  # instead of loss.backward()
                optimizer.step()


Lite(...).run(args)
```

This is how you would call the function

```python
Lite(strategy="ddp", devices=8, accelerator="gpu", precision="bf16").run(10)
```

## What's in this repo?

### Example with `flair`

There are three `py` scripts under `example-flair`. You have:

- `run_ner.py`: original model training script, for reference.
- `custom_run_ner.py`: modified training script with an integration to LightningLite.
- `custom_trainer.py`: modified version of `trainer.py` containing commented changes to the original code to add compatibility with `PL`.

Please start from `example-flair/notebook_flair.ipynb`.

Main changes to the original `trainer.py`:

```python
# model definition
self.model, optimizer = self.setup(self.model, optimizer)
```

```python
# setting up dataloaders
batch_loader = self.setup_dataloaders(batch_loader)
```

I've been having issues with how batches are being split across GPUs so I had to go back to the original model functions to calculate the forward loss. This should be revised. Perhaps [this issue](https://github.com/flairNLP/flair/issues/499) gives a good idea about what's happening.

```python
# forward pass
loss = self.model_original.forward_loss(batch_step) # work in progress // to be replaced
```

```python
# backward prop
self.backward(loss)
```

Main changes to the original `run_ner.py`:

```python
@dataclass
class TrainingArguments:
    num_epochs: int = field(default=10, metadata={"help": "The number of training epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size used for training."})
    mini_batch_chunk_size: int = field(
        default=1,
        metadata={"help": "If smaller than batch size, batches will be chunked."},
    )
    learning_rate: float = field(default=5e-05, metadata={"help": "Learning rate"})
    seed: int = field(default=42, metadata={"help": "Seed used for reproducible fine-tuning results."})
    device: str = field(default="cuda:0", metadata={"help": "CUDA device string."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for optimizer."})
    embeddings_storage_mode: str = field(default="none", metadata={"help": "Defines embedding storage method."})
    # adding new arguments below
    accelerator: Optional[str] = field(default=None, metadata={"help": "Choose the hardware to run on e.g. 'gpu'."})
    strategy: Optional[str] = field(
        default=None, 
        metadata={"help": "Strategy for how to run across multiple devices e.g. 'ddp', 'deepspeed'."})
    devices: Optional[int] = field(
        default=None, 
        metadata={"help": "Number of devices to train on (int), which GPUs to train on (list or str)"})
    num_nodes: Optional[int] = field(default=1, metadata={"help": "Number of GPU nodes for distributed training."})
    precision: Optional[int] = field(default=32, metadata={"help": "Choose training precision to use."})
```

```python
    # changed
    trainer = LiteTrainer( 
        accelerator=training_args.accelerator,
        strategy=training_args.strategy,
        devices=training_args.devices,
        num_nodes=training_args.num_nodes,
        precision=training_args.precision,
    )
    
    # changed
    trainer.train(tagger, corpus,
        data_args.output_dir,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.batch_size,
        mini_batch_chunk_size=training_args.mini_batch_chunk_size,
        max_epochs=training_args.num_epochs,
        embeddings_storage_mode=training_args.embeddings_storage_mode,
        weight_decay=training_args.weight_decay,
    )
```

### Example with `HuggingFace`

In order to make it easier to run benchmarks against other NLP frameworks such as [`stanza`](https://stanfordnlp.github.io/stanza/), [`PaddleNLP`](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/README_en.md#quick-start), etc., I placed an example of how to run distributed training on `SageMaker` using `HuggingFace`. It uses `run_ner.py` as `entry_point` and automatically install all required libraries. You can see the file under `example-huggingface/notebook_huggingface.ipynb`.

The codebase shows how to get a `flair` corpus (`NER_ENGLISH_PERSON`) converted to json and re-uploaded into AWS S3 for further training.

## Disclaimer
- The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
- The ideas and opinions outlined in these examples are my own and do not represent the opinions of AWS.