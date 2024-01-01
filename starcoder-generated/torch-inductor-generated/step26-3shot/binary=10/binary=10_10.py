
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.other = torch.nn.Parameter(torch.randn(4))
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1
        v3 = v2.transpose(0, 1).view(v2.shape[1]) + self.other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(4, 3)
