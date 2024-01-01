
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=1)
        self.act_4 = torch.nn.LeakyReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v2 = self.conv_transpose(x)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 13, 13)
