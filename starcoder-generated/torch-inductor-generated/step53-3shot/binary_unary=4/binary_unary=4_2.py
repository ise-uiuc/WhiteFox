
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(224, 256)
        self.other = other

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(1, 224, 224))

# Inputs to the model
x = torch.randn(1, 224, 224)
