
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.transpose(x1, 0, 1)
        x4 = torch.transpose(x1, 0, 1)
        x5 = torch.transpose(x2, 0, 1)
        return torch.cat([x5, x4, x3], 0)
# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(3)
