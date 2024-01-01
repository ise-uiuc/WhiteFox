
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 128, 1, stride=2, padding=0, dilation=1, output_padding=0)
        self.fc = torch.nn.Linear(768, 1024, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.flatten(1)
        v3 = self.fc(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
