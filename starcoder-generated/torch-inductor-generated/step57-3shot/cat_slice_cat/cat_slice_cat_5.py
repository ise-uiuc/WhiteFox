
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, xList):
        x = torch.cat(xList, dim=1)
        x = x[:, 0:9223372036854775807]
        x = x[:, 0:np.random.choice(28)]
        x = torch.cat([x, xList[1]], dim=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7, 28, 28)
x2 = torch.randn(1, 7, 28, 28)
