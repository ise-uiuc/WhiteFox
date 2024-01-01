
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        v1 = torch.nn.Conv3d(2, 2, 2)
        v2 = torch.reshape(v1.weight, (2, 2))
        v3 = torch.reshape(v2, (1, 2, 2))
        self.weight = v3
    def forward(self, x):
        x = torch.reshape(x, (1, 2, 2))
        x = torch.matmul(x, self.weight)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
