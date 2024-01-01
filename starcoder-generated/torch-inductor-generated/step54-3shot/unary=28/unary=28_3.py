
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256, bias=False)
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=0.3)
        v3 = torch.clamp(v2, max=0.5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
