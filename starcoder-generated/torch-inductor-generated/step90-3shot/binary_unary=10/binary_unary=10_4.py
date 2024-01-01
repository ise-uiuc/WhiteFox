
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.randn(256, 21504)
        b1 = torch.zeros(256)
        self.linear = torch.nn.Linear(21504, 256, bias=True)
        self.linear.weight = torch.nn.Parameter(w1)
        self.linear.bias = torch.nn.Parameter(b1)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + x_input # where x_input is another variable
        v3 = F.relu(v2)
        return v3

# Initialize the model
m = Model()

# Inputs to the model
x_input = torch.randn(1, 256)

