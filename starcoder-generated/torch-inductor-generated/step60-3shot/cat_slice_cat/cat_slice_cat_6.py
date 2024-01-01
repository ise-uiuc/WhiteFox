
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.concat_dim1 = torch.nn.Conv2d(10, 1, 1, stride=1, padding=0, bias=True)
  
    def forward(self, x1, x2):
        x = [x1, x2]
        x1 = x[0]
        x2 = x[1]
        x = torch.cat(x, dim=1)
        x = x[:, 0:9223372036854775807]
        x = x[:, 0:size]
        x = torch.cat([x1, x2], dim=1)
        return self.concat_dim1(x)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
x2 = torch.randn(1, 10, 64, 64)
