
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6,3)
    def forward(self, x):
        y = torch.sigmoid(self.linear(x))
        z = torch.sigmoid(self.linear(x))
        u = self.linear(y)
        v = self.linear(z)
        return u,v
# Inputs to the model
x = torch.randn(2, 6)
