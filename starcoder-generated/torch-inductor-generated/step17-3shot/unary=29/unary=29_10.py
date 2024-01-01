
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=5.6):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 1, stride=1, padding=1)
        self.act_4 = torch.nn.ReLU6()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x4):
        v5 = self.conv_transpose(x4)
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        v8 = self.sigmoid(v7)
        return v8
# Inputs to the model
x4 = torch.randn(1, 3, 224, 224)
