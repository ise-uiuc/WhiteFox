
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 * torch.clamp(t1+3, max=6)
        t3 = t2 / 6
        return t3
# Initializing the model
d1 = Model()
# Inputs to the model
x1 = torch.randn(1, 16)
print(d1(x1))

