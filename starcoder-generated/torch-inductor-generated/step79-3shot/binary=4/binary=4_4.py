
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 100)
        self.linear2 = torch.nn.Linear(20, 100)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 20)
x3 = torch.randn(1, 20)
x4 = torch.randn(1, 20)
x5 = torch.randn(1, 20)
x6 = torch.randn(1, 20)
x7 = torch.randn(1, 20)
x8 = torch.randn(1, 20)
x9 = torch.randn(1, 20)
x10 = torch.randn(1, 20)
x11 = torch.randn(1, 20)
x12 = torch.randn(1, 20)
