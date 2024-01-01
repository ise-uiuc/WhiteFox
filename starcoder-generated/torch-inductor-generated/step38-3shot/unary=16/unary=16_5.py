
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(192, 256)
        self.linear.weight = torch.nn.Parameter(torch.zeros(self.linear.weight.shape))
        self.linear.bias = torch.nn.Parameter(torch.zeros(self.linear.bias.shape))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 192)
