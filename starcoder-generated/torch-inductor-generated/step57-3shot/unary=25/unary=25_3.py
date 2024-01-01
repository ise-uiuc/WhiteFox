
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_slope = torch.neg(0.01) # This is the negative slope
        self.linear = torch.nn.Linear(10, 20) # This is a linear transformation module with 10 input features and 20 output features
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.neg_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
