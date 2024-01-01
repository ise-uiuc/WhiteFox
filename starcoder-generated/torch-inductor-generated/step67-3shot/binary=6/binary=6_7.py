 with one line difference from the first model example

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(x1)
        v3 = v1 + v2
        return v3
```
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
