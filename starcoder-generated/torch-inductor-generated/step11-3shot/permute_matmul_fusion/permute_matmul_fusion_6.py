
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.bmm(x1.permute(0, 2, 1), x2.permute(0, 2, 1)).permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 32, 2, 256)
x2 = torch.randn(1, 32, 256, 2)
