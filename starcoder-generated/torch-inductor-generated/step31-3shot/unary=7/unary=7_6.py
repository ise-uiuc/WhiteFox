
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        self.min = torch.tensor(0.0)
        self.max = torch.tensor(6.0)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 * (v1.clamp(min=self.min, max=self.max) + self.min) / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 16)
