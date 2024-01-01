
class Model(torch.nn.Module):
    def __init__(self, o):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.other = o
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 2)
o1 = torch.randn(1, 2)
