
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        return v1[:, 0:9223372036854775807, 0:6]

# Initializing the model
m = Model()

# Inputs to the model
size = torch.randint(1, 5, [1])
x1 = torch.randn(3, 7, 5)
x2 = torch.randn(3, 7, 5)
