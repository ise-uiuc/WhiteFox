
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.where(v1 > 0, v1, v1 * negative_slope)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 128)
