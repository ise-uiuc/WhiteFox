 
class MyModel(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = torch.nn.Conv2d(1,32,3,1,0)
      self.conv2 = torch.nn.Conv2d(32,64,3,1,0)
      self.conv3 = torch.nn.Conv2d(64,128,3,1,0)
      self.conv4 = torch.nn.Conv2d(128,256,3,1,0)
      self.conv5 = torch.nn.Conv2d(256,512,3,1,0)
      self.conv6 = torch.nn.Conv2d(512,512,3,1,0)
      self.linear = torch.nn.Linear(10,100)
      self.linear2 = torch.nn.Linear(100,10)
      self.relu = torch.nn.ReLU()
      self.max_pool2d = torch.nn.MaxPool2d(2)
      self.dropout = torch.nn.Dropout(0.5)
      self.flatten = torch.nn.Flatten()
      self.sigmoid = torch.nn.Sigmoid()
      self.avg = torch.nn.AdaptiveAvgPool2d((1,128))


  def forward(self, x):
      x = self.conv1(x)
      x = self.relu(x)
      x = self.conv2(x)
      x = self.relu(x)
      x = self.max_pool2d(x)
      x = self.dropout(x)
      x = self.conv3(x)
      x = self.relu(x)
      x = self.conv4(x)
      x = self.relu(x)
      x = self.max_pool2d(x)
      x = self.dropout(x)
      x = self.conv5(x)
      x = self.relu(x)
      x = self.conv6(x)
      x = self.relu(x)
      x = self.max_pool2d(x)
      x = self.flatten(x)
      x = self.linear(x)
      x = self.relu(x)
      x = torch.cat((x,0.5*x),1)
      x = self.linear2(x)
      x = self.sigmoid(x)
      x = self.avg(x)
      x = self.softmax(x)
    
      return x

# Instantiating the model 
model = MyModel()

# Model inputs 
x = torch.randn(512,3,10,10)

# The result of forwarding the model with the given input 
result = model(x)
