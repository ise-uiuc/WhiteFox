
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 10, bias=False)
        self.other = torch.nn.Parameter(torch.randn(10))
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
