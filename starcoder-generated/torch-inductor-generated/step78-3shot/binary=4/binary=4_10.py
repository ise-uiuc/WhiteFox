
class Model(torch.nn.Module):
    def __init__(self, n1, n2, n3):
        super().__init__()
        self.linear1 = torch.nn.Linear(n1, n2)
        self.linear2 = torch.nn.Linear(n2, n3)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1) + x2
        return v2

# Initializing the model
m = Model(4, 5, 3)

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 3)
