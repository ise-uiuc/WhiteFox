2
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc_0 = torch.nn.Linear(4, 4)
    self.fc_7 = torch.nn.Linear(4, 4)
    self.fc_8 = torch.nn.Linear(4, 4)
  def forward(self, x):
    v1 = torch.relu(self.fc_0(x))
    v4 = self.fc_7(v1)
    v8 = self.fc_8(v1)
    v9 = torch.pow(v4, v8)
    v10 = torch.nn.functional.log(v9)
    return v10
# Inputs to the model
x = torch.randn(1, 4)
