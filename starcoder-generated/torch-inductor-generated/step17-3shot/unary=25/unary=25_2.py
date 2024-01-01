
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 > 0
        w3 = w1 * self.negative_slope
        w4 = torch.where(w2, w1, w3)
        return w4

# Initializing the model
m = Model(0.01)

# Inputs to the model
x1 = torch.randn(32, 16)
