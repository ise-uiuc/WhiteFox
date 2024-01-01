
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, (1, 1), bias=True, padding=(0, 0), stride=(1, 1))
    self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 8, (1, 1), bias=True, padding=(0, 0), stride=(1, 1))
  def forward(self, x1):
    v1 = self.conv_transpose(x1)
    v2 = torch.relu(v1)
    v3 = self.conv_transpose1(v2)
    v4 = torch.relu(v3)
    return v4
# Input to the model
x1 = torch.randn(1, 1, 3, 5)
