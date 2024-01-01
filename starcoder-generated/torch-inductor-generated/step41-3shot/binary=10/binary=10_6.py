
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + torch.tensor([-1.0, -0.7071067811865476, -0.5, 0.5, 0.7071067811865476])
        return v2

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
