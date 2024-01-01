
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor):
        v1 = torch.nn.Linear(1000, 10)
        v2 = v1(input_tensor) - 500
        v3 = torch.relu(v2)
        return v3

```
# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(2, 1000)
