
class Model(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, N)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.relu(v1)
        return v2

# Initializing the model
m = Model(10, 28 * 28)

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
