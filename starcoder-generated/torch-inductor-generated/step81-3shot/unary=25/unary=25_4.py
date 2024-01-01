
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=16, out_features=15, bias=True)
        self.negative_slope = negative_slope

    def set_negative_slope(new_negative_slope):
        self.negative_slope = new_negative_slope

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = (v1 > 0)
        v3 = v2 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 16)
