
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.negative_slope = negative_slope
    
    def forward(self, x1):
        h1 = self.linear(x1)
        h2 = h1 > 0
        h3 = h1 * self.negative_slope
        h4 = torch.where(h2, h1, h3)
        return h4
    
# Initializing the model
m = Model(negative_slope=0.01)

# Inputs to the model
x1 = torch.randn(4, 16)
