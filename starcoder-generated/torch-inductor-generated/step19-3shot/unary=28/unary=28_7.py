
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16, bias=True)
 
        self.min_value = torch.nn.Parameter(torch.tensor(0.1))
        self.max_value = torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
