
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4, bias=False)
        self.other = torch.nn.Parameter(torch.Tensor([[1.0, 2.0, -3.0, 5.0, -6.0, 4.0]]))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 6)
