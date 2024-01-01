
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = F.linear(x1, torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([1.0]))
        t2 = v1 > 0
        t3 = v1 * self.negative_slope
        t4 = torch.where(t2, v1, t3)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
