
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + x
        return v2
 
# Initializing the model
m = Model2()
# Inputs to the model
x = torch.randn(1, 5)
