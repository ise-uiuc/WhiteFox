
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 24, bias=False)
        self.min_value = torch.tensor(0.0)
        self.max_value = torch.tensor(6.0)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
