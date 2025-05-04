simulated distributed training

#### Example

<details>
<summary>train a timm model on a torchvision dataset</summary>

```python
class TimmWrapper(nn.Module):
    def __init__(self, model_id: str, num_classes: int, pretrained: bool) -> None:
        super(TimmWrapper, self).__init__()
        self.model = timm.create_model(
            model_id, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

transform = transforms.Compose([transforms.ToTensor()])
ds = CIFAR100(root="artifacts", train=True, transform=transform, download=True)

engine = DiLoCo(
    model_cls=TimmWrapper,
    model_kwargs={
        "model_id": "timm/resnet18.tv_in1k",
        "num_classes": len(ds.classes),
        "pretrained": True,
    },
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={},
    outer_optimizer_cls=torch.optim.SGD,
    outer_optimizer_kwargs={"lr": 0.7, "nesterov": True, "momentum": 0.9},
    train_dataset=ds,
    criterion=F.cross_entropy,
    batch_size=64,
    eval_steps=200,
    num_nodes=8,
    num_epochs=1,
    warmup_steps=0,
    diloco_interval=500,
    wandb_kwargs={
        "project": "diloco",
        "entity": "sauravmaheshkar",
    },
)
engine.fit()
```

</details>

#### Implementations

* [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)

#### References

* https://github.com/matttreed/diloco-sim
* https://github.com/jianbo27/diloco-sim
