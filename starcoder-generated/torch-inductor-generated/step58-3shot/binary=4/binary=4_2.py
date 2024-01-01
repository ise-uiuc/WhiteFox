
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
 
    def forward(self, x):
        v = self.linear(x)
        m = torch.nn.ReLU()
        return m(v)

# Initializing the model
m = Model()
# Inputs to the model
x = torch.randn(1, 32)
