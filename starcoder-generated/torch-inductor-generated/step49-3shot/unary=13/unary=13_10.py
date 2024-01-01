
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x2):
        b1 = self.linear(x2)
        b2 = torch.sigmoid(b1)
        b3 = b1 * b2
        return b3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 16)
