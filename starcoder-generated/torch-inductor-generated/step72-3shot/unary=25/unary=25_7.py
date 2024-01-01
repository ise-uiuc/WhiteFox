
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.permute(0, 2, 3, 1).reshape(int(x1.shape[0]), -1, int(x1.shape[1]))
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = v4.reshape(int(x1.shape[0]), int(x1.shape[1]), int(x1.shape[2]), int(x1.shape[3])).permute(0, 3, 1, 2)
        return v5

# Initializing the model
m = Model()
m.negative_slope = 0.01

# Inputs to the model
x1 = torch.randn(1, 32, 32, 3)
