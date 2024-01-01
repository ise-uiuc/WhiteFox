
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * 0.5
        y3 = y1 + (y1 * y1 * y1) * 0.044715
        y4 = y3 * 0.7978845608028654
        y5 = torch.tanh(y4)
        y6 = y5 + 1
        y7 = y2 * y6
        return y7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
