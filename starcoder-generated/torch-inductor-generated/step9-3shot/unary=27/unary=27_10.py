 # Note: change requires_grad
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 9, 2, stride=2, padding=3)
        self.min = min
        self.max = max
    def forward(self, input):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        # Note: change requires_grad
        v3.requires_grad_()
        return v3
min = 0.1
max = 0.4
# Inputs to the model
input = torch.randn(1, 4, 200, 100)
