
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=2):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.pad1 = torch.nn.ReflectionPad2d((0, 1, 0, 1))
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 16, 2, stride=2, padding=0)
        self.pad2 = torch.nn.ReflectionPad2d((1, 2, 1, 2))
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(16, 3, 2, stride=2, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v4 = self.pad1(x1)
        v1 = self.conv_transpose(v4)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v5 = self.pad2(v3)
        v6 = self.conv_transpose_1(v5)
        v7 = self.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
