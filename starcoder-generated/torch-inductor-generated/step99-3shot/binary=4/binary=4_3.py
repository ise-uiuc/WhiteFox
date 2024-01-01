
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10, bias=True)
 
    def forward(self, x1, x2):
        y1 = self.linear(x1)
        y2 = y1 + x2
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(3, 5)
x2 = torch.rand(3, 10)
y = m(x1, x2)

