
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 16)
        self.linear2 = torch.nn.Linear(16, 16)
 
    def forward(self, x, h1, h2):
        v1 = self.linear1(x)
        v2 = v1 * v2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
h1 = torch.randn(1, 16)
h2 = torch.randn(1, 16)
