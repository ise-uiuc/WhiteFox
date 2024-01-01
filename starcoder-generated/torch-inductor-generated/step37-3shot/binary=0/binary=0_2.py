
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x1, other=None):
    for i in range(0,3):
      x1 = torch.sum(x1, i)
    return x1
# Inputs to the model 
x1 = torch.randn(1, 2, 5)
