
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        b1 = torch.cat([x1[:, :, 0:4611686018427387903],x1[:, :, 4611686018427387903:9223372036854775807]], dim=2)
        b2 = b1[:, :, 0:9223372036854775807]
        b3 = b2[:, :, 0:6]
        return b3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 9223372036854775807)
