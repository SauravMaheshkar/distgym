import os
import warnings

import torch
from input_pipeline import prepare_datasets
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

from distgym.core.diloco import DiLoCo


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


class LlamaAlpacaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaForCausalLM(config)

    def forward(self, x):
        device = next(self.model.parameters()).device
        input_ids = x[:, 0, :].long().to(device)
        attention_mask = x[:, 1, :].long().to(device)
        labels = x[:, 2, :].long().to(device)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss.view(1)


class LlamaWrapper(DiLoCo):
    def inner_step(self):
        ## Fetch batch
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.epoch += 1
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.rank == 0:
            if self.wandb is not None:
                self.wandb.log(
                    {
                        "loss": loss.item(),
                        "perplexity": torch.exp(loss).item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
            else:
                print(
                    f"Step {self.local_step}, Loss: {loss.item()}, Perplexity: {torch.exp(loss).item()}"  # noqa: E501
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    seq_length = 1024

    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path="config.json")

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_datasets(tokenizer, max_length=seq_length)

    engine = LlamaWrapper(
        model_cls=LlamaAlpacaModel,
        model_kwargs={"config": config},
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 4e-4, "weight_decay": 0.1, "betas": (0.9, 0.95)},
        outer_optimizer_cls=torch.optim.SGD,
        outer_optimizer_kwargs={"lr": 0.7, "nesterov": True, "momentum": 0.9},
        train_dataset=dataset,
        batch_size=32,
        num_nodes=1,
        num_epochs=1,
        warmup_steps=1000,
        diloco_interval=500,
        wandb_kwargs={
            "project": "diloco",
            "entity": "sauravmaheshkar",
            "tags": ["llama-baseline"],
        },
    )
    engine.fit()
