
class LRelu(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.weight, self.bias)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
    
m = LRelu(0.5)

# Inputs to the model
x1 = torch.randn(1, 32, 4, 4)
