
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        negative_slope =.2
        return torch.where(v1 > 0., v1, v1 * negative_slope)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
