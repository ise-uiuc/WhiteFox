
class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(48,16, bias=True)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1, other=x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 48)
x2 = torch.randn(1, 48)
