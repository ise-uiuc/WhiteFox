
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
        # "w" is an attribute of torch.nn.Linear, which is the weight matrix in the linear transformation
        # "weight_orig" is the original weight matrix in the linear transformation
        # Because "w" is a weight matrix and "weight_orig" is an attribute of torch.nn.Linear, they should have equal values.
        self.linear.weight_orig = torch.nn.Parameter(self.linear.weight.detach().clone())
        self.linear.bias.data[:] = 0

    def forward(self, x):
        t1 = self.linear(x)
        t2 = t1 + self.linear.weight_orig
        return torch.relu(t2)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
