
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(36, 10)
 
    def forward(self, x1):
        x2 = torch.reshape(x1, [-1, 36])
        x3 = self.linear(x2)
        x4 = torch.tanh(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1296)
