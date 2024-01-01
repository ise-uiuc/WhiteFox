
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v8 = x1.permute(0, 2, 1)
        v9 = x2.permute(0, 2, 1)
        return torch.bmm(v8, v9)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
