
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 8, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
x2 = torch.randn(1, 8)
