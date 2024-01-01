
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16*5*5, 512)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(x1.size(0), -1))
        v2 = v1 - 1000.
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16 * 5 * 5)
