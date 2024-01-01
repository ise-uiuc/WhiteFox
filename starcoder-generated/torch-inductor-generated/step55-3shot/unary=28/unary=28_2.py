
class Model(torch.nn.Module):
    def __init__(self, linear_min_value, linear_max_value, clamp_min_value, clamp_max_value):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)
        self.clamp_min = torch.nn.ReLU6()
        self.clamp_max = torch.nn.Softshrink()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v2, self.linear_min_value)
        v3 = self.relu6(v2, self.linear_max_value)
        v4 = self.shrink(v3, self.clamp_min_value)
        return v4

# Initializing the model
m = Model(-1, 20, 40, 50)

# Inputs to the model
x1 = torch.randn(1, 10)
