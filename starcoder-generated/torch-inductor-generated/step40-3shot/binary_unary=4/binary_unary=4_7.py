
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 128)
 
    def forward(self, **x1):
        v1 = self.linear(x1["input"])
        v2 = v1 + x1["other"]
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 28 * 28)
o1 = torch.randn(10, 28 * 28)
