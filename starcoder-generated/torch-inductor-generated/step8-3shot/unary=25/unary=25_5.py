
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10, bias=False)
        self.negative_slope = 0.02
 
    def forward(self, x1):
        v1 = self.linear(x1)
        positive_mask = (v1>0).type_as(v1)
        negative_mask = (v1<=0).type_as(v1)
        v2 = self.linear.weight * self.negative_slope
        v3 = torch.where(positive_mask, v1, v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(20,5)
