
class Model(torch.nn.Module):
    def __init__(self, *, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(128 * 128, 256)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = torch.clamp_min(t1, self.min_value)
        t3 = torch.clamp_max(t2, self.max_value)
        return t3

# Initializing the model
m = Model(min_value=-1, max_value=1)

# Inputs to the model
x1 = torch.randn(1, 128 * 128)
