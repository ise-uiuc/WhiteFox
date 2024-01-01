
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 > 0
        v3 = v1 * args.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Inputs to the model
args.negative_slope = 0.01

x2 = torch.randn(1, 3)
