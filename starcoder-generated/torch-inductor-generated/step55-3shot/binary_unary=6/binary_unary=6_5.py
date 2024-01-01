
class Model(torch.nn.Module):
    # Other member functions omitted.

    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 - other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 100)
other = torch.randn(1)
