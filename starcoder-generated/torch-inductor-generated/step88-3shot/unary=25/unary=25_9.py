
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 6)
 
  def forward(self, x1, slope):
    v1 = self.linear(x1)
    v2 = v1 > 0
    v3 = v1 * slope
    v4 = torch.where(v2, v1, v3)
    return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
slope = torch.randn(1, )
