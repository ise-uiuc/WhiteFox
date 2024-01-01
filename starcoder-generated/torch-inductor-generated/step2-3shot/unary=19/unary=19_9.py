
class Model(torch.nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.linear = torch.nn.Linear(n1, n2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
n1, n2, n3 = 12, 13, 14
m = Model(n1, n2)

# Inputs to the model
x1 = torch.randn(10, n1)
