
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return torch.cat([x, x, x], dim=1)
# Inputs to the model
x = torch.randn(3, 2, 4)
