
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        return torch.cat([x1, x2, x3, x4], dim=1)[:, 0:9223372036854775807][:, 0:8257]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 64, 25)
x2 = torch.randn(4, 64, 20)
x3 = torch.randn(4, 64, 40)
x4 = torch.randn(4, 128, 25)
