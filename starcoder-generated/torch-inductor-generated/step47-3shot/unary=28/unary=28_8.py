
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=False)
 
    def forward(self, x, min_value=0., max_value=1.):
        x1 = self.fc(x)
        x2 = torch.clamp_min(x1, min_value)
        x3 = torch.clamp_max(x2, max_value)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
