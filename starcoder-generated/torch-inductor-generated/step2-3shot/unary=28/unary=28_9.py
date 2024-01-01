
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        o1 = torch.clamp_min(v1, self.min_value)
        o2 = torch.clamp_max(o1, self.max_value)
        return o2

# Initiali
min_value, max_value = 0, 2
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3)
