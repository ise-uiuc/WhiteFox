
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1, w1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.linear(v1, w1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
w1 = torch.randn(20, 10)
