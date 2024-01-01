
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=32, out_features=64, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1 + 3, 0, 6)
        return v2 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
