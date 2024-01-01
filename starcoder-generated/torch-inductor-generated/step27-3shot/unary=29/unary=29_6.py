
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.3, max_value=-0.1):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.act_4 = torch.nn.Tanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.tanh(v3)
        v9 = self.act_4(v4)
        return v9
# Inputs to the model
x2 = torch.randn(1, 3, 1, 5)
