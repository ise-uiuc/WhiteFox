
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v):
        return torch.cat([v, v], 1)
# Inputs to the model
v = torch.randn(4, 4)
