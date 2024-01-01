
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=4, out_features=2, bias=False)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3


# Initializing the model
m = Model()
# Inputs to the model
x = torch.randn(1, 4)
