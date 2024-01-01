
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_features = 32
        self.linear = torch.nn.Linear(3, num_features, bias=False)

    def forward(self, x, other=None):
        v1 = self.linear(x)
        v2 = v1 + other
        y = F.relu(v2)
        return y


# Initializing the model
num_features = 32
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
other = torch.randn(1, num_features).sigmoid()
