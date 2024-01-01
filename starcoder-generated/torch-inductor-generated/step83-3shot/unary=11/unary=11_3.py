
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.sin(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sin(v2)
        v4 = torch.cos(v3)
        v5 = torch.sin(v4)
        v6 = torch.tensor([3.0], dtype=torch.float32)
        v7 = v5 + v6
        v8 = torch.clamp_max(v6 + v7,6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
