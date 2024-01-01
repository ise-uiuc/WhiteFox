
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1, minV=0, maxV=0.5):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, minV)
        v3 = torch.clamp_max(v2, maxV)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
