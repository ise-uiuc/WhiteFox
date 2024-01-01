
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1[0] - 4
        v3 = v1[1] - 4
        v4 = torch.max(v2, v3)
        v5 = v1[0] * v1[1] * v4
        return v5

# Initialize the model
m = Model()

# Inputs to the model
x1 = torch.randn(2)
