
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = torch.addmm(x1, self.linear.weight, self.linear.bias)
        v2 = torch.cat((v1), 3)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 1, 1)
