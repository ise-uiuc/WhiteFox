
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.9, max_value=-1.8):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(1, 16, 1, stride=1, padding=0)
        self.act_1 = torch.nn.ReLU6()
        self.act_2 = torch.nn.ELU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = self.act_1(v1)
        v3 = v2 / 0.132624187636261
        v4 = self.act_2(v3)
        v5 = v4 - 0.03400395462517737
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
