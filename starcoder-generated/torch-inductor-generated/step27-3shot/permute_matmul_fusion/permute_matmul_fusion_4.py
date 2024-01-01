
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x3, x4):
        v2 = torch.bmm(x3, x4)
        v3 = v2.permute(0, 2, 1)
        return v2
# Inputs to the model
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(1, 2, 2)
