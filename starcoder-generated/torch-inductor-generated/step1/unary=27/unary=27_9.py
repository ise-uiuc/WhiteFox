
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.min_value = torch.nn.Parameter(torch.Tensor([0]))
        self.max_value = torch.nn.Parameter(torch.Tensor([1]))
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
