
```
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)

    def forward(self, x1: torch.Tensor, other: torch.Tensor):
        x2 = self.linear(x1)
        x3 = x2 + other
        out = x3 + 1
        return out
```
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
other = torch.randn(2)
