
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
    def forward(self, x2):
        v0 = x2
        v1 = v0.flatten(1, 2)
        if torch.allclose(v1[:, :, 0], v1[:, :, -1]):
            v1 = v1 + torch.randn(1, 2, 2)
        v2 = v1.reshape(-1, 1, 4, 2)
        v3 = v2 + torch.randn(1, 1, 4, 2)
        v4 = v3.reshape(-1, 2, 2, 2)
        v5 = v4.permute(0, 2, 3, 1)
        v6 = v5.flatten(1, 2)
        v7 = torch.cat((v6[:, :, 0::2], v6[:, :, 1::2]), 2)
        return v7
# Inputs to the model
x2 = torch.randn(1, 1, 2, 8)
