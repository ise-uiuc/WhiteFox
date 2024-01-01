
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(2, 1, 0)
        v2 = torch.bmm(v1, x2)
        v3 = v2.permute(2, 0, 1).contiguous()
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
