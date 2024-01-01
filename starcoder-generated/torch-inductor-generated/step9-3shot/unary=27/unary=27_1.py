
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 13, 3, stride=1, padding=7)
        self.min = min
        self.max = max
    def forward(self, input_tensor):
        v1 = self.conv(input_tensor)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.03277192257352356
max = 0.01987902042524
# Inputs to the model
input_tensor = torch.randn(1, 4, 128, 128)

