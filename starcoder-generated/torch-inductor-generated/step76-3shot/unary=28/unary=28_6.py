
class Model(torch.nn.Module):
    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.linear = torch.nn.Linear(256, 512)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_min(v2, self.max_value)
        return v3

# Initializing the model
m1 = Model(0, 255)
m2 = Model(-255, -1)

# Inputs to the model
x1 = torch.randn(1, 256)
__output1__ = m1(x1)
__output2__ = m2(x1)

