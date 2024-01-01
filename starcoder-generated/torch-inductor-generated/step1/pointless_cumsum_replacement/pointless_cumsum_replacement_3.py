
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(80, 10)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.cumsum(torch.f32(torch.full((80,), 1)), 1)
        return v1 * v2

# Initialzing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 80)
