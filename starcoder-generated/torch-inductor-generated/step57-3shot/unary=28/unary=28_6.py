
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m1 = Model(min_value=(1, 1), max_value=(2, 2))

# Inputs to the model
x1 = torch.randn(1, 2, 3)
__output1__ = m1(x1)

# Other values that min_value and max_value can take
m2 = Model(min_value=-1, max_value=-2)
__output2__ = m2(x1)
m3 = Model(min_value=0, max_value=0)
__output3__ = m3(x1)
m4 = Model(min_value=2, max_value=1)
__output4__ = m4(x1)

