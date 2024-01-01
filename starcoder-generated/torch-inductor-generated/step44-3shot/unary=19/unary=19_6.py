
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Input tensor for the model
x2 = torch.randn(1, 32)
