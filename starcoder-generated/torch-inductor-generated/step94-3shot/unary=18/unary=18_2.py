
class Model(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.conv = torch.nn.Conv2d(32, 32, (3, 5), stride=(1, 3), padding=(1,2))
      self.relu = torch.nn.ReLU(inplace=True)
   def forward(self, x3):
      v0 = self.conv(x3, )
      v1 = self.relu(v0, )
      return v1
# Inputs to the model
x3 = torch.randn(1, 32, 8, 59)
