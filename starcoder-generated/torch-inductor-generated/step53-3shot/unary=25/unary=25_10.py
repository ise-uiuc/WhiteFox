
class Model(torch.nn.Module):
  def __init__(self, negative_slope: float):
      super().__init__()
      self.linear = torch.nn.Linear(3, 8, bias=False)
      self.negative_slope = negative_slope

  def forward(self, x1):
      v1 = self.linear(x1)
      v2 = v1.tolist()
      for i in range(len(v2)):
          for j in range(len(v2[i])):
              if v2[i][j] > 0:
                  v2[i][j] = v2[i][j] * self.negative_slope
              else:
                  v2[i][j] = v2[i][j]
      v3 = torch.tensor(v2)
      v4 = torch.where(v1 > 0, v1, v3) # v1 > 0 bool tensor, v3tensor
      return v4

# Initializing the model
m = Model(negative_slope=0.25)

# Inputs to the model
x1 = torch.randn(1, 3)
