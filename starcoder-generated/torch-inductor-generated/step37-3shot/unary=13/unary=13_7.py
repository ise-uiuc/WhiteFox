
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = torch.nn.Linear(3, 8)
        self.linear_1 = torch.nn.Linear(8, 3)
 
    def forward(self, x_0):
        v1 = self.linear_0(x_0)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        o = self.linear_1(v3)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
