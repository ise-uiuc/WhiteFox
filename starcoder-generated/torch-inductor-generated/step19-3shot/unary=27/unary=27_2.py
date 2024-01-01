
class Model(torch.nn.Module):
    def __init__(self, min=753.0, max=-8162.0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(9, 256, kernel_size=5, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=5)
        self.min, self.max = min, max
    
    def forward(self, input):
        v0 = self.conv1(input)
        v1 = self.conv2(v0)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
# Inputs to the model
input = torch.randn(1, 9, 100, 100)
