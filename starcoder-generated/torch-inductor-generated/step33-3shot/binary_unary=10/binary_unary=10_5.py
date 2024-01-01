
```
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
        v3 = torch.relu(v2)
        return v3
```

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 3)
