
class ModelTanh(torch.nn.Module):
    def __init__(self):
      super(ModelTanh, self).__init__()
      self.conv1 = torch.nn.Conv2d(1, 6, 3, padding=1)
      self.conv2 = torch.nn.Conv2d(6, 12, 3,padding=1)
      self.conv3 = torch.nn.Conv2d(12, 24, 3,padding=1)
      self.pool = torch.nn.MaxPool2d(2, 2)
      self.dropout = torch.nn.Dropout2d(0.25)
    def forward(self, x):
      x = x.view(-1,1,28,28)
      x = torch.tanh(self.conv1(x))
      x = self.pool(x)
      x = self.dropout(x)
      x = self.conv2(x)
      x = torch.tanh(x)
      x = self.pool(x)
      x = self.dropout(x)
      x = self.conv3(x)
      x = torch.tanh(x)
      x = self.pool(x)
      x = self.dropout(x)
      x = x.view(-1,8)
      x = torch.tanh(x)
      return x
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
