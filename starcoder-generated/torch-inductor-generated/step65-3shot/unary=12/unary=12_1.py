
class Model(torch.nn.Module):
  def __init__(self):
      super(Model, self).__init__()
      self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
      self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
      self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
      self.fc2 = torch.nn.Linear(500, 10)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4 * 4 * 50)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x
# Model end
# inputs to the model
x1 = torch.randn((1, 1, 28, 28))
# Use the generated model.
m1 = Model()
# Print out the generated model's structure.
print(m1)
# Print the output of passing the inputs to the model.
print(m1(x1).size())

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Use padding and dilation to generate new valid PyTorch models. Please be creative when using these parameters.
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, dilation=1)


    def forward(self, inputs):
        output = self.conv1(inputs)
        return output

m1 = Model()
# Print out the generated model's structure.
print(m1)