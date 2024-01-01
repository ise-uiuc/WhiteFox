
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = torch.nn.functional.interpolate(v1, scale_factor=0.5, recompute_scale_factor=True)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randint(255, (1, 3, 64, 64))
