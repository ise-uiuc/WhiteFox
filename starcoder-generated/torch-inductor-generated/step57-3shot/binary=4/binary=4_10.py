
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 5, bias=True)
 
    def forward(self, x1):
        return self.linear(x1) + torch.randn(5, 2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2)
