
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = torch.nn.Linear(64, 64)
        self.linear_1 = torch.nn.Linear(64, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear_0(x1) + x2
        v2 = v1 + x2
        v3 = self.linear_1(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
