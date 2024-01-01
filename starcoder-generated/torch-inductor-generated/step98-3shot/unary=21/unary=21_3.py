
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch_uu.nn_uu.Conv2d(1, 1, 1)
  def forward(self, x):
        x1 = self.tanh(self.conv1(x))
        return x1
# Inputs to the model
x = torch.randn(1, 1, 128)
