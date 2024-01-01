
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.9984045963287354, max_value=0.9661341428756714):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        v3 = self.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
