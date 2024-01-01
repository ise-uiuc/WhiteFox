
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, size):
        v1 = torch.cat((x1, x2), dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 62, 62)
x2 = torch.randn(1, 5, 62, 62)
size = 3
