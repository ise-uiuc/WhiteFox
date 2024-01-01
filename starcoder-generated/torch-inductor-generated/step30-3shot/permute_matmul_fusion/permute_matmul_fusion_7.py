
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(2, 1, 0)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.bmm(x1, x2)
        return (v1, v2, v3)
# Inputs to the model
x1 = torch.randn(2, 2, 1)
x2 = torch.randn(1, 2, 2)
