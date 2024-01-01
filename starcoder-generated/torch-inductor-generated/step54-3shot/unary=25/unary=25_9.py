 initialization
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        v4 = torch.where(t2, t1, t3)
        return v4

# Input to the model
x1 = torch.randn(1, 64)
negative_slope = 0.1
