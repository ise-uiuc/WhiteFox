
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 4)

  def forward(self, X):
    return self.linear(X) + torch.randn(4)

# Initializing the model
m = Model()

# Inputs to the model
X = torch.randn(1, 10) # 1 is the batch size, 10 is the dimension of the input
