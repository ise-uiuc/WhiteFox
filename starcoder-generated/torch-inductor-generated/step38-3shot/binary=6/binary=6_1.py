
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2



The model's linear transformation's input tensor shape is `(32,)`, while the shape of the `other` tensor is `(32, 32)`, the model will meet this pattern when being used as following:
```
model()
```

# Inputs to the model
x1 = torch.randn(32)
other = torch.randn(32, 32)
