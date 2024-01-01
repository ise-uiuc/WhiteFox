
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(6, 6)
 
    def forward(self, x1):
        x2 = x1.view(x1.size(0), -1)
        return torch.clamp_max(torch.clamp_min(self.block(x2), 1), -1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
