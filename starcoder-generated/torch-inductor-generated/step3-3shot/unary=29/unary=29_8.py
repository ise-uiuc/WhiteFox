
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=1):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.convt(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        v3 = torch.clamp(v2, self.min_value, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
