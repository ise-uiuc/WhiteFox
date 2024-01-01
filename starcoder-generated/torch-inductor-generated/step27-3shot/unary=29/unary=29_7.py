
class Model(torch.nn.Module):
    def __init__(self, min_value=2.6, max_value=6.9):
        super().__init__()
        self.transpose_conv2d_1 = torch.nn.ConvTranspose2d(2, 10, 1, stride=1, padding=0)
        self.act_2 = torch.nn.ReLU6()
        self.transpose_conv2d_2 = torch.nn.ConvTranspose2d(10, 10, 1, stride=1, padding=0)
        self.act_5 = torch.nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.transpose_conv2d_1(x2)
        v2 = v1 + -2.648787352965254
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.act_2(v4)
        v6 = self.transpose_conv2d_2(v5)
        v7 = torch.clamp_min(v6, self.min_value)
        v8 = torch.clamp_max(v7, self.max_value)
        v9 = self.act_5(v8)
        return v9
# Inputs to the model
x2 = torch.randn(1, 2, 224, 224)
