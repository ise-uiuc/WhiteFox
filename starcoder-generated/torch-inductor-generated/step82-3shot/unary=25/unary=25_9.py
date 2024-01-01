
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.zeros((x1.shape[1], x1.shape[1])), bias=None)
        positive_mask = (v1 > 0).float()
        negative_mask = (v1 < 0).float()
        negative_slope = self.negative_slope
        v3 = (1 + negative_slope) * (v1 * positive_mask)
        v4 = v3 + v3 * negative_sage * negative_mask 
        return v4

# Initializing the model
m = Model(-0.01)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
