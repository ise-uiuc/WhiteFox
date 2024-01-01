
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input, other):
        return self.linear(input) + other

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, in_features)
other = torch.randn(1, out_features)
