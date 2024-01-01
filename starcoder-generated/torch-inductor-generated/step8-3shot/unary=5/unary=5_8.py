
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 2, 3, stride=3, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 1, 2, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose2(v6)
        v8 = self.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 33, 33)
