
class Model(torch.nn.Module):
    def __init__(self, min_value=45.49, max_value=176.31):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(5, 5, 3, stride=2, padding=2, dilation=3)
        self.act_0 = torch.nn.ReLU6()
        self.act_3 = torch.nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v10 = self.conv2d(x1)
        v12 = torch.clamp_min(v10, self.min_value)
        v15 = self.act_0(v12)
        v16 = torch.clamp_max(v15, self.max_value)
        v18 = self.act_3(v16)
        return v18
# Inputs to the model
x1 = torch.randn(1, 5, 8, 8)
