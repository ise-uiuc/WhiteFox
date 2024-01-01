
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Conv2d(8*8*8, 16, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp((torch.clamp((v1, min=self.min_value), max=max_value)), min=self.min_value)

# Initializing the model
m = Model(min_value=-1, max_value=2)

# Inputs to the model
x1 = torch.randn(1, 8*8*8)
