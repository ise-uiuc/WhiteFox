
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v, _ = torch.sort(x1, 1)
        v = v[:, -5:]
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(3, 4)
