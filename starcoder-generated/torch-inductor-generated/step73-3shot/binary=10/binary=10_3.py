
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1.add(other)
        return v2

# Initializing the model
m = Model()

# Input1 to the model
x1 = torch.randn(32, 128)
# Input2 to the model
x2 = torch.randn(32, 128)
