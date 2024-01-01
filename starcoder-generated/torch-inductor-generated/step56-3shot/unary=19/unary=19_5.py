
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5, bias=False)
        self.linear.weight.data = torch.randn(5, 10)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.sigmoid(v1)
        v3 = v2.mean(dim=1)
        return v3

# Initializing the model
m2 = Model2()

# Inputs to the model
x2 = torch.randn(10, 10)
__output2__ = m2(x2)

