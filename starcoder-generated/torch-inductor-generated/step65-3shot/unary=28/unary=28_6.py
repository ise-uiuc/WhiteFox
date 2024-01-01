
class Model(torch.nn.Module):
    def __init__(self, minimum, maximum):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.minimum = minimum
        self.maximum = maximum
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp(min=self.minimum)
        v3 = v2.clamp(max=self.maximum)
        return v3

# Initializing the model
m = Model(minimum=-0.1, maximum=0.5)

# Inputs to the model
x1 = torch.randn(1, 4)
