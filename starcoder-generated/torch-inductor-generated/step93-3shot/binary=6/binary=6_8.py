
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1) - x2
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn((1), requires_grad=True)
