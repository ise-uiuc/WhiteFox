
class ResnetBlock(torch.nn.Module):
    def __init__(self, dim):
      super(ResnetBlock, self).__init__()
      self.conv_block = torch.nn.Sequential(
          torch.nn.ReplicationPad2d(1),
          torch.nn.Conv2d(dim, dim, 3))
    def forward(self, x1):
      return x1 + self.conv_block(x1)
# Inputs to the model
input = torch.randn(1, 3, 224, 224)
