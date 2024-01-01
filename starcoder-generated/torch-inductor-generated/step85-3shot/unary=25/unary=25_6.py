
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        negative_slope = torch.nn.Parameter(np.array(-0.1, dtype=np.float32))
        return torch.where(v1 > 0, v1, v1 * negative_slope)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
