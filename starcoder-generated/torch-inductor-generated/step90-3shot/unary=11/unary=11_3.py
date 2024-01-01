
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 64, 3, stride=2)
        self.add = torch.nn.Add()
        self.clamp1 = torch.nn.Hardtanh()
        self.clamp2 = torch.nn.Hardtanh()
        self.divide = torch.nn.Hardtanh()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.add(v1, 3)
        v3 = self.clamp1(v2)
        v4 = self.clamp2(v3)
        v5 = self.divide(v4 / 6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6)
