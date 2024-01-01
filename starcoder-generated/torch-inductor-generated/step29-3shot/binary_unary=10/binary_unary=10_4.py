
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(20, 4)
        self.other = torch.from_numpy(other)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(np.random.randn(20).astype(np.float32))

# Inputs to the model
x1 = torch.randn(1, 20)
