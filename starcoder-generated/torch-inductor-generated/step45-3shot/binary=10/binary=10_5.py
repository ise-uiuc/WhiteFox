
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8)

    def forward(self, x, other):
        v1 = self.fc1(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 5)
__other = torch.randn(10, 8)
