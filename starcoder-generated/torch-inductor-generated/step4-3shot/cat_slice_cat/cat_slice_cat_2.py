
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat((x1, x2, x3), 1)
        v2 = v1[:, :9223372036854775807]
        v3 = v2[:, :192]
        return torch.cat((v1, v3), 1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 153, 76, 76)
x2 = torch.randn(1, 187, 140, 140)
x3 = torch.randn(1, 180, 140, 140)
