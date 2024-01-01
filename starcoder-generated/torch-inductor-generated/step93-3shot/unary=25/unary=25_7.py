
class Model(torch.nn.Module):
    def __init__(self, negativeSlope):
        super().__init__()
        self.negativeSlope = negativeSlope
        self.linear = torch.nn.Linear(8, 24)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        negativeSlope = self.negativeSlope
        v3 = v1 * negativeSlope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(negativeSlope=-0.3)

# Inputs to the model
x1 = torch.randn(1, 8)
