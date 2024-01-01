
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
        
for negative in [negative for negative in np.arange(0.001, 0.1, 0.001)]:
    m = Model(negative)
    # Inputs to the model
    x1 = torch.randn(1, 8)
    __output_diff__ = m(x1)
