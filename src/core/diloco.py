import math
import os
from copy import deepcopy
from typing import Callable, Type

import torch
import torch.distributed as dist


class DiLoCo:
    def __init__(
        self,
        model_cls: Type[torch.nn.Module],
        model_kwargs: dict,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        outer_optimizer_cls: Type[torch.optim.Optimizer],
        outer_optimizer_kwargs: dict,
        train_dataset: torch.utils.data.Dataset,
        criterion: Callable[..., torch.Tensor],
        batch_size: int,
        eval_steps: int,
        num_nodes: int,
        num_epochs: int,
        warmup_steps: int,
        diloco_interval: int,
        cosine_anneal: bool = False,
        wandb_kwargs: dict = None,
    ) -> None:
        super().__init__()

        ## Model Attributes
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs

        ## Optimizer Attributes
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.outer_optimizer_cls = outer_optimizer_cls
        self.outer_optimizer_kwargs = outer_optimizer_kwargs

        ## Dataset Attributes
        self.train_dataset = train_dataset

        ## Training Attributes
        self.criterion = criterion
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.num_nodes = num_nodes
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.diloco_interval = diloco_interval
        self.cosine_anneal = cosine_anneal
        self.max_local_step = (
            self.num_epochs
            * len(self.train_dataset)
            // (self.batch_size * self.num_nodes)
        )
        self.local_step: int = 0
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.wandb_kwargs = wandb_kwargs

    def initialize(self, rank: int) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12355"
        self.rank = rank

        ## Initialise the process group and set up devices
        dist.init_process_group(
            backend=(
                "nccl"
                if torch.cuda.is_available()
                and self.num_nodes == torch.cuda.device_count()
                else "gloo"
            ),
            rank=self.rank,
            world_size=self.num_nodes,
        )
        self.device = torch.device(
            f"cuda:{self.rank % torch.cuda.device_count()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        torch.cuda.set_device(self.device) if self.device.type == "cuda" else None

        ## Initialise the model
        self.model = self.model_cls(**self.model_kwargs).to(self.device)
        for _, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)
        self.model.train()

        ## Initialise the optimizer
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), **self.optimizer_kwargs
        )

        ## Initialise the scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(self.warmup_steps, 1))
            elif self.cosine_anneal:
                progress = (current_step - self.warmup_steps) / float(
                    max(1, self.max_local_step - self.warmup_steps)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                return 1.0  # Default constant LR after warmup

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

        ## Initialise the dataset
        sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=self.num_nodes,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            pin_memory=True,
        )
        self.train_iter = iter(self.train_loader)

        ## Outer Process
        if self.rank == 0:
            ## wandb
            if self.wandb_kwargs is not None:
                try:
                    import wandb

                    self.wandb = wandb
                    self.wandb_kwargs = self.wandb_kwargs
                    self.wandb.login()
                except ImportError:
                    pass

            if self.wandb is not None:
                self.wandb.init(
                    project=self.wandb_kwargs["project"],
                    entity=self.wandb_kwargs["entity"],
                    config=self.config,
                )

            self.outer_model = deepcopy(self.model).to("cpu")
            for param in self.outer_model.parameters():
                param.requires_grad = True

            self.outer_optimizer = self.outer_optimizer_cls(
                self.outer_model.parameters(), **self.outer_optimizer_kwargs
            )

    def flush(self):
        if self.rank == 0:
            if self.wandb is not None:
                self.wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

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
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
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

    def outer_step(self):
        ## avg params
        for param in self.model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.num_nodes

        if self.rank == 0:
            self.outer_optimizer.zero_grad()

            ## outer gradient
            for name, param in self.model.named_parameters():
                param.grad = (
                    self.outer_model.state_dict()[name].data.to(param.device)
                    - param.data
                )

            self.outer_optimizer.step()

            ## sync outer model
            for name, param in self.outer_model.named_parameters():
                param.data = self.model.state_dict()[name].data.to("cpu")

        ## broadcast params
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def train(self, rank: int):
        try:
            self.initialize(rank)

            while self.local_step < self.max_local_step:
                self.inner_step()

                if self.local_step % self.diloco_interval == 0:
                    self.outer_step()

                self.local_step += 1

        finally:
            self.flush()

    def fit(self):
        torch.multiprocessing.spawn(
            self.train, args=(), nprocs=self.num_nodes, join=True
        )
