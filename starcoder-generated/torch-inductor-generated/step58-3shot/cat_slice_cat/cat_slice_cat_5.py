
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x = x1 + x2
        y = x[0, 0:9223372036854775807]
        x = torch.cat([x, y], dim=1)
        return x
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
x2 = torch.randn(1, 10, 32, 32)
