
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=0)
        x = x[:, 0:9223372036854775807]
        x = x[:, 0:min(x1.size(1), x2.size(1))]
        x = torch.cat([x1, x], dim=1)
        return x
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 280, 121)
x2 = torch.randn(1, 10, 460, 199)
