
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2[:, :, :, ::-1]
        v4 = v3[:, :, :, ::-1]
        v5 = v3[:, :, :, ::-1]
        v6 = v4[:, :, :, ::-1]
        v7 = v5[:, :, :, ::-1]
        v8 = v7[:, :, :, ::-1]
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
