
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.Tensor(2, 2)
        self.bias = torch.Tensor(1)
 
    def forward(self, x1):
        v1 = torch.addmm(x1, self.weight, self.bias)
        v2 = torch.cat((v1,), 0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
