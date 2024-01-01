
class Model(torch.nn.Module):
    def __init__(self, negative_slope: float):    
        super().__init__()
 
        self.negative_slope = negative_slope

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.ones_like(x1), bias=torch.zeros_like(x1))
        v2 = v1 > 0
        v3 = self.negative_slope * v1
        v4 = torch.where(v2, v1, v3)
        return v4

# Instantiation of the model
m = Model(negative_slope=0.2)

# Inputs to the model
x1 = torch.randn(2, 3, 4)
