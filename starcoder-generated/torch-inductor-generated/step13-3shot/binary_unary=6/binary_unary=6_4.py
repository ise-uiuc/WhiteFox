
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 24)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        o1 = math.sqrt(1 / v1.shape[1])
        v2 = v1 - o1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
