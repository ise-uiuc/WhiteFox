
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16 * 16 * 3, 10)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        return y1

# Initializing the model
m = Model()

# Inputs of the model
x1 = torch.randn(1, 16 * 16 * 3)
y1 = torch.randn(1, 10)
