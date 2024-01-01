
class Model(torch.nn.Module):
    def __init__(self, size0, size1):
        super().__init__()
        self.linear = torch.nn.Linear(size0, size1, bias=False)
        self.other = torch.rand(size1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
size0 = 5
size1 = 10
m = Model(size0, size1)

# Input to the model
x1 = torch.rand(1, size0)
