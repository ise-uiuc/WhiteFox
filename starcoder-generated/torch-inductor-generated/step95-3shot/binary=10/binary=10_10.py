
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 12)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return v1 + other

 
m = Model()

# Inputs to the model
## The "other" input is a tensor with shape [1, 3]
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 3)
