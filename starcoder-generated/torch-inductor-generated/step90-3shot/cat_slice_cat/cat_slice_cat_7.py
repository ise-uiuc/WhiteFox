
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, 16)
        self.layer2 = torch.nn.Linear(16, 1)

    def forward(self, x1):
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = torch.cat((x2, x3), dim=1)
        x5 = x4[:, 0:9223372036854775807]
        x6 = x5[:, 0:9223372036854775807]
        x7 = torch.cat((x4, x6), dim=1)
        return x7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
