
class Model(torch.nn.Module):
    def __init__(self, min_value=0.01, max_value=0.25):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
        self.relu = torch.nn.ReLU(True)
    def forward(self, x1):
        v1 = self.convt(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
