
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        __self.linear1__ = torch.nn.Linear(3, 12)
        __self.linear2__ = torch.nn.Linear(12, 16)
 
    def forward(self, x1):
        v1 = self.linear1_(x1)
        v2 = self.linear2_(v1)
        v3 = v2 + other
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 10)
other = torch.randn(1, 16, 10)
