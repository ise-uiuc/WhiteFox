
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=4, out_features=8)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

for i in [-3, 3]:
    # Initializing the model with the specified negative slope
    m = Model(i)

    # Inputs to the model
    x1 = torch.randn(1, 4)
    