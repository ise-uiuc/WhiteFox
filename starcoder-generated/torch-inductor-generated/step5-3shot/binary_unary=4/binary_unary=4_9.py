
class Model(torch.nn.Module):
    def __init__(self, other):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(other=1)

# Input to the model
x1 = torch.randn(1, 3, 64, 64)
