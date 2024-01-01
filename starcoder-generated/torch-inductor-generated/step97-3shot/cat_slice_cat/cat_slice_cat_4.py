
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = x[:, :18446744073709551615]
        x = x[:, :x.shape[2] / 2]
        y = torch.cat([x1, x], dim=1)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 100, 1000)
x2 = torch.randn(2, 100, 2000)
x3 = torch.randn(2, 100, 2000)
