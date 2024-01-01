
class Model(torch.nn.Module):
    def forward(self, x):
        a1 = x[0]
        a2 = x[1]
        a3 = torch.cat([a1, a2], dim=1)
        a4 = a3[:, 0:9223372036854775807]
        a5 = a4[:, 0: 32]
        a6 = torch.cat([a3, a5], dim=1)
        return a6

# Initializing the model
m = Model()

# Inputs to the model
x = [
    torch.randn(1, 8, 32, 32),
    torch.randn(1, 4, 32, 32)
]
