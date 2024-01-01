
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        y = self.linear(x1)
        y1 = torch.sigmoid(y)
        y2 = y * y1
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
y = m(x1)

