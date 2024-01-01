
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = x[:, 0: 9223372036854775807]
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 20)
