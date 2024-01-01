
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat(x1, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:5]
        v4 = torch.cat(v1, v3, dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = [torch.randn(1, 7, 2, 3),
     torch.randn(1, 9, 2, 9),
     torch.randn(1, 5, 5, 3)]
