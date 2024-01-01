
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v7 = self.linear(x1)
        v8 = v7 + 1
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
other = torch.FloatTensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
