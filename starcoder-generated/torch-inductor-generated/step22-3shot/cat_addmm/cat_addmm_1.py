
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Linear(2, 3)
        self.t = nn.Linear(2, 3)
    def forward(self, x): # Add self.m and self.t as inputs.
        x = self.m(x)
        x = self.t(x)
        x = torch.stack([x, x], dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
