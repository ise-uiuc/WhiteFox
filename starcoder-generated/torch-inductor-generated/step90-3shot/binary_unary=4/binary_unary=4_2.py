
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(29, 29)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3
# Initializing the model
other = torch.randn(29, 29)
m = Model(other)

# Input to the model
x = torch.randn(29, 29)
