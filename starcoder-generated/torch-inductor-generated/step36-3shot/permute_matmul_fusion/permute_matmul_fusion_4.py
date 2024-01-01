
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1).squeeze(-1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.nn.functional.conv2d(v2, v1)
        return v3.squeeze(-1)
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
x2 = torch.randn(1, 2, 4, 4)
