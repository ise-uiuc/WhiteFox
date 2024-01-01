
class Model(torch.nn.Module):
    def __init__(self, other=None):
        super().__init__()
        if other is None:
            other = torch.empty([2, 5])
        self.linear = torch.nn.Linear(6, 5)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        if self.other is not None:
            v1 = v1 + self.other
        return v1

# Initializing the model
m = Model()
x1 = torch.randn(2, 6)
