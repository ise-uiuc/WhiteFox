
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=1.0):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.softsign = torch.nn.Softsign()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.relu6(v3)
        v5 = self.softsign(v4)
        return torch.flatten(v5, 1)
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
