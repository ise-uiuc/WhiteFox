
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# A second tensor to be concatenated with the output of the linear transformation
x2 = torch.randn(1, 3, 64, 64)

# Initializing the model
m = Model()

# Inputs to the model
