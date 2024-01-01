
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16 * 16 * 2, 10)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
