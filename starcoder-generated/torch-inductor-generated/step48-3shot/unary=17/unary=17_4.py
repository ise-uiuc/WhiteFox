
class Conv_ReL(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.block0 = torch.nn.Sequential(torch.nn.Conv2d(3, 31, (3, 5), bias=False, padding_mode='zeros'), torch.nn.ReLU(inplace=True))
  def forward(self, x1):
      v1 = self.block0(x1)
      v2 = torch.argmax(v1, dim=1)
      v3 = torch.max(v1, dim=1)[1]
      v4 = torch.flatten(torch.mul(v2, v3))
      v5 = torch.nn.functional.pad(v2, (4,), "constant", 12)
      return torch.transpose(torch.bmm(v4, v5), 0, 1)

model = Conv_ReL().apply(conv_to_fc)
