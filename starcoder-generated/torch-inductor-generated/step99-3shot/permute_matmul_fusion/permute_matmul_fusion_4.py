
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(2, 0, 1)
        v2 = x2.permute(2, 1, 0)
        v12 = torch.matmul(v1, v2)
        v3 = v12.permute(2, 1, 0, 3, 4)
        v4 = v12[:, :, 0] - v12[:, :, 2] - v12[:, :, 3] + v12[:, :, 5]
        return v4.flatten()
# Inputs to the model
x1 = torch.randn(4, 1, 3)
x2 = torch.randn(4, 1, 3)
