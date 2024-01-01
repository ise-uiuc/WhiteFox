
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1[v1 > 0]
        v3 = v1[v2 < 0] * -self.negative_slope
        v4 = torch.where(v1 > 0, v1, v3)
        return v4
 
# Initializing the model with negative slope = 0.01
m = Model(0.01)

# Inputs to the model
x1 = torch.randn(1, 8)
