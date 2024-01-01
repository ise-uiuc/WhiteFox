
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, X):
        Y = torch.nn.functional.linear(X, self.weight, bias)
        Y_mask = Y!= 0
        return torch.where(Y_mask, Y, -self.negative_slope * Y)

# Initializing the model
m = Model(negative_slope=0.2)

# Inputs to the model
X = torch.randn(1, 3)
