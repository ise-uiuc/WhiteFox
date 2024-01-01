
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(1, stride=1, padding=6) # Replace max with any function call that only takes the first argument as input, or change this call to a function which takes multiple arguments
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0
max = 1
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
