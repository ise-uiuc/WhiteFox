
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(23 * 23 * 3, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + torch.tanh(x2)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 23 * 23 * 3)
x2 = torch.randn(1, 11 * 11 * 3)
