
class Model(torch.nn.Module):
    def forward(self, x1, other):
        return torch.nn.functional.linear(x1, other)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(8, 64, 64)
