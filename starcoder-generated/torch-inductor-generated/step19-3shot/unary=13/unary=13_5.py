 architecture
class GatingConv(torch.nn.Module):
    def __init__(self, linear, sigmoid):
        super().__init__()
        self.linear = linear
        self.sigmoid = sigmoid
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = self.sigmoid(y1)
        y3 = y1 * y2
        return y3

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GatingConv(torch.nn.Linear(2, 8), torch.nn.Sigmoid())
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = self.sigmoid(y1)
        y3 = y1 * y2
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
