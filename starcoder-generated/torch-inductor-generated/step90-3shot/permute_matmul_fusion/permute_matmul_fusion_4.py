
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(1, 0)
        v2 = x2.permute(1, 0)
        v3 = torch.bmm(x2, v1)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(2, 2)
