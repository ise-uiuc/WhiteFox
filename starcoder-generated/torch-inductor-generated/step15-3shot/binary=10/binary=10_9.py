
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __a = __output__ + torch.randn(1, 1)
        v3 = torch.tanh(__a)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
