
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, size):
        v1 = torch.cat((x1, x2), 1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat((v1, v3), 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(2, 8, 1, 11)
x2 = torch.rand(2, 5, 1, 6)
v1 = torch.randint(1, 8, (1,))
