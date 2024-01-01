


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv_transpose3 = torch.nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv_transpose2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv_transpose3(v4)
        v6 = v5 + 3
        v7 = torch.clamp(v6, min=0)
        v8 = torch.clamp(v7, max=6)
        v9 = v5 * v8
        v10 = v9 / 6
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
