
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10, bias=True)

    def forward(self, x1):
      v1 = self.linear(x1)
      v2 = v1 > 0
      v3 = v1 * 0.29884204188062289739276747770763095404888
      v4 = torch.where(v2, v1, v3)
      return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
