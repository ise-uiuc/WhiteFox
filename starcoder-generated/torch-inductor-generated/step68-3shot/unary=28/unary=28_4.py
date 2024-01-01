
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=256, out_features=256, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_max(torch.clamp(v1, min=-6.0), max=6.0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
