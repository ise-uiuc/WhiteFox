
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.cat((x, x), 1)
        v2 = torch.cat((v1, v1), 1)
        return torch.cat((v1, v2[1:]))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 2, 1)
