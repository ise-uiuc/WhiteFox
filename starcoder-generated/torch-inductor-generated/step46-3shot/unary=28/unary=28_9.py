
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32)
 
    def forward(self, x1):
        y = self.linear(x1)
        y = torch.clamp_min(y, min=0)
        y = torch.clamp_max(y, max=6000)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
