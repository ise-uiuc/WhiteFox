
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.min_tensor = torch.tensor(min)
        self.max_tensor = torch.tensor(max)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self._clamp_tensor_min(v1)
        v3 = self._clamp_tensor_max(v2)
        return v3
    def _clamp_tensor_min(self, x1):
        v1 = torch.clamp_min(x1, self.min_tensor)
        return v1
    def _clamp_tensor_max(self, x1):
        v1 = torch.clamp_max(x1, self.max_tensor)
        return v1
min = 0.3
max = 0.3
# Inputs to the model
x1 = torch.randn(1, 3, 52, 52)
