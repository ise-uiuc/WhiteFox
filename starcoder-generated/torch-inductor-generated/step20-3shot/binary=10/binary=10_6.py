
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8, in_features = 3, bias = True)
        self.linear2 = torch.nn.Linear(8, 8, in_features = 8, bias = True)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8, 32, 32)
