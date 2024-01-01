
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x1, x2, x3):
        x4 = x1 + x2
        x5 = torch.cat((x3, x4), dim=1)
        v1 = self.linear(x5)
        v2 = torch.addmm(v1, v1, v1)
        return v2


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
x3 = torch.randn(17, 4)
