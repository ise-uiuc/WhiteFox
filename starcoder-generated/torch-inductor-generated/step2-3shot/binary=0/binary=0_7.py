
class Model(torch.nn.Module): # PyTorch 1.6.0
 def __init__(self):
  super().__init__()
      self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
 
 def forward(self, x1):
  v1 = self.conv(x1)
  v2 = v1 + other
  return v2

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

