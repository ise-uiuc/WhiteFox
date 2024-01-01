
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = torch.tanh(v7)
        return v8


# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
__output__2 = m2(x2)