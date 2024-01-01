
class Model(torch.nn.Module):
    def forward(self, x):
        a = torch.cat(x)
        b = a[:, 0:9223372036854775807]
        c = b[:, 0:0]
        return torch.cat([a, c])

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(3, 4)
x3 = torch.randn(4)
x4 = torch.randn(5)
