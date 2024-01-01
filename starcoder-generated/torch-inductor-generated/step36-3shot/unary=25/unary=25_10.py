
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = x1.flatten(start_dim=1)
        v2 = torch.nn.functional.linear(v1, torch.tensor([[1.78790387, -0.01625918, 0.01750951, 1.26051818, -0.49119654, -0.50248609, -0.50834412, -0.00280192]]))
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3 > 0, v2, v4)
        v6 = torch.reshape(v5, (2, 8))
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
