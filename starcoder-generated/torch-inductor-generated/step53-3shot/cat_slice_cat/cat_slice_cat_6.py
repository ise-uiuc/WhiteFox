
class Model(torch.nn.Module):
    def forward(self, x, y):
        t = torch.cat([x, y], dim=1)
        t = t[:, 0:9223372036854775807]
        t = t[:, 0:25]
        t = torch.cat([x, y], dim=1)
        return t

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 50, 100)
y = torch.randn(1, 75, 100)
