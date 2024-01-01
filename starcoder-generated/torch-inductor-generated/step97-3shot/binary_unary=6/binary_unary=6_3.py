
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = self.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
