
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v2 = torch.softmax(x1, dim=-1)
        v3 = torch.softmax(v2, dim=-1)
        v4 = v2.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 3)
x2 = torch.randn(1, 3, 3)
