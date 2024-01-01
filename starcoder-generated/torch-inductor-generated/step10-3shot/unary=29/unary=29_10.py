
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=2):
        super(Model, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, 1, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        x2 = self.conv_transpose2d(x1)
        x3 = torch.clamp_min(x2, self.min_value)
        x4 = torch.clamp_max(x3, self.max_value)
        x5 = self.tanh(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
