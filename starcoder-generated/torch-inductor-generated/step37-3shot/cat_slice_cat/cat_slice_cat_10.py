
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
      y1 = torch.cat((x1, x2, x3, x4), dim=1)
      y2 = y1[:, :9223372036854775807]
      y3 = y2[:, :1]
      return torch.cat((y1, y3), dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 1, 64)
x2 = torch.randn(100, 1, 64)
x3 = torch.randn(100, 1, 64)
x4 = torch.randn(100, 1, 64)
x5 = torch.randn(100, 1, 64)
