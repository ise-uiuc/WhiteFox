
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min_tensor = torch.tensor([min])
        self.max_tensor = torch.tensor([max])
        self.conv = torch.nn.Conv2d(5, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_tensor)
        v3 = torch.clamp_max(v2, self.max_tensor)
        return v3
min = 0.41
max = 0.05
# Inputs to the model
x1 = torch.randn(1, 5, 20, 20)
