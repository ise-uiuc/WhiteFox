
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v2 = torch.bmm(x2, v1)
        v3 = torch.bmm(x1, v2)
        return torch.bmm(x1, v1)

# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(10, 2, 2)
