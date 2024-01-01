
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 1, 2)
        v2 = x1.permute(0, 2, 1)
        v3 = torch.bmm(v1, v2)
        v4 = torch.bmm(v3, v1)
        return (v3, v4)
# Inputs to the model
x1 = torch.randn(1, 3, 2)
x2 = torch.randn(1, 2, 5)
