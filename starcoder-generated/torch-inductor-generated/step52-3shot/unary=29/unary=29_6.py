
class Model(torch.nn.Module):
    def __init__(self, min_value=-5.3, max_value=5.3):
        super().__init__()
        self.act_3 = torch.nn.LeakyReLU(0.7)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v4 = self.act_3(x3)
        v6 = torch.clamp_min(v4, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        return v7
# Inputs to the model
x3 = torch.randn(1, 3, 30, 59)
