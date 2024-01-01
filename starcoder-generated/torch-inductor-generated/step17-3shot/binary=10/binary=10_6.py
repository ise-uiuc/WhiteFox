
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32)
 
    def forward(self, x2):
        v1 = self.linear1(x2)
        v2 = v1 + 0.1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
other = torch.randn(1, 32)
