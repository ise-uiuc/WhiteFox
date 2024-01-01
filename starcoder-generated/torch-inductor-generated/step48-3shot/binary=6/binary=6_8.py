
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 12)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v1, v2

# Initializing and testing the model
m = Model()
m.eval()
x1 = torch.randn(4, 20)
x2 = torch.randn(20)
v1, v2 = m(x1, x2)
print(torch.equal(v1, v2))

