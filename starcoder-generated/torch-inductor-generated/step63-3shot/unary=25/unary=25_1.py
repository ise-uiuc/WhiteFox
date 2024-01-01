
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 50)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 >= 0
        v3 = v2 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)

# Taken from: https://discuss.pytorch.org/t/leaky-relu-not-working/2232/8
m.negative_slope = -0.01

