 definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 64)
        self.linear2 = torch.nn.Linear(64, 1)
 
    def forward(self, x1, other):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = self.linear2(t2)
        return v3


# Inputs to the model
x1 = torch.randn(1, 16)
