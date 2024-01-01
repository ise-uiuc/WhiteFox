
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torch.nn.Linear(2, 2, bias=False)
 
    def forward(self, x1):
        v1 = self.transform(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, requires_grad=True)
