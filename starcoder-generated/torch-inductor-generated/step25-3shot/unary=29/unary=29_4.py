
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.5, max_value=4.9):
        super(Model, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.tanh(v3)
        return v4

# Inputs to the model
x2 = torch.randn(1, 1, 56, 56)
