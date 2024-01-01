
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.gt(v1, torch.tensor(0))
        v3 = torch.mul(v1, torch.tensor(self.negative_slope))
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([[0.7513]])
