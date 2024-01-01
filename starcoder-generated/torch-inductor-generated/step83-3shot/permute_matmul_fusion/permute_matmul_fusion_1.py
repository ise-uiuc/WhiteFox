
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        v2 = torch.bmm(v1, v3).transpose(2, 1)
        v4 = torch.bmm(v3, v2)
        v5 = torch.bmm(v2, v3)
        v6 = torch.bmm(v2, v4)
        return (v2, v5, v6)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
