
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20, bias=False)
 
    def forward(self, x):
        return torch.tanh(self.linear(x))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5, requires_grad=True)
