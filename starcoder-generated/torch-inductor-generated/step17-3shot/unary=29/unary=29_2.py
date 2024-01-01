
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.3, max_value=3.3):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.act_3 = torch.nn.ReLU6()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x5):
        v4 = self.conv_transpose2d(x5)
        v6 = torch.clamp_min(v4, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        v9 = self.act_3(v7)
        return v9
# Inputs to the model
x5 = torch.randn(1, 3, 3, 3)
