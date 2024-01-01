
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.conv_transpose = nn.ConvTranspose2d(4, 10, 1, stride=1, padding=1)
        self.act_12 = nn.ReLU6()
        self.min_value = min_value
    def forward(self, x3):
        v1 = self.conv_transpose(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = self.act_12(v2)
        return v3
# Inputs to the model
x3 = torch.randn(1, 4, 224, 224)
