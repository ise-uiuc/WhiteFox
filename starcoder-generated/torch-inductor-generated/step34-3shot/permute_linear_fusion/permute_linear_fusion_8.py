
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.cat((v1, v1, v1, v1, v1), 2)
        v3 = v2.squeeze(-1)
        v4 = v3.view((-1, 16))
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
