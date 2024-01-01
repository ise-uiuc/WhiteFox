
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v3 = torch.bmm(x2.permute(0, 2, 1), x1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
