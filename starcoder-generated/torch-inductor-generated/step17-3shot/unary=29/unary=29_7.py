
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.3, max_value=1.3):
        super(Model, self).__init__()
        self.softsign = torch.nn.Softsign()
        self.batch_norm = torch.nn.BatchNorm2d(16)
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 16, 1, stride=1, padding=1)
        self.act_10 = torch.nn.LeakyReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x4):
        v1 = self.conv_transpose(x4)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.softsign(v3)
        v9 = self.act_10(v4)
        return v9
# Inputs to the model
x5 = torch.randn(1, 10, 124, 124)
