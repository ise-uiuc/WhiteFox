
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(0, 2, 1)
        v1 = x2.permute(0, 2, 1)
        v2 = v0[0][0]
        v3 = v1[0][0]
        v4 = torch.bmm(v2.unsqueeze(0), v3.unsqueeze(1))
        return v4.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
