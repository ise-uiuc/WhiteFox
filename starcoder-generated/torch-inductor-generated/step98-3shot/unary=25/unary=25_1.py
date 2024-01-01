
class Model(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear = torch.nn.Linear(int(params['in_features']), int(params['out_features']), bias=False)
        self.negative_slope = float(params['negative_slope'])
        
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3) 
        return v4

# Initializing the model
m = Model({"in_features" : 256, "out_features" : 32, "negative_slope" : -0.25})

# Inputs to the model
x1 = torch.randn(1, 256)
__output_1__ = m(x1)

