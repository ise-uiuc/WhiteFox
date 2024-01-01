
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(1, 0, 2)
        v1 = x1.permute(1, 0, 2)
        v3 = torch.bmm(x1.permute(1, 0, 2), x2.permute(1, 0, 2))

        output = (v1 + v2) * (v3 + v4)
        return output
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
