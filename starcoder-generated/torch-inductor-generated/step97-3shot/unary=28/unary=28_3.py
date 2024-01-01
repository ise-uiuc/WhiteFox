
class Model(torch.nn.Module):
    def forward(self, x1, min_value=-1.0, max_value=100.0):
        v1 = torch.nn.functional.linear(x, self.weight, bias)
        v2 = torch.max(v1, min_value)
        v3 = torch.min(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(32, 100)

# Specifying the values of the arguments
min_value = -10.0
max_value = 10.0

