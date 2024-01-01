
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(1,1), stride=1)
        torch.manual_seed(2)
        torch.nn.init.constant_(self.conv.weight,0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.squeeze(v1, dim=2)
        v3 = torch.squeeze(v2, dim=1)
      return v3
# Inputs to the model
x1 = torch.randn(1, 3, 44, 75)
