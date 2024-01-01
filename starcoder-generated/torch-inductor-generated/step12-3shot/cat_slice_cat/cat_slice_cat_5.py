
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1[0], x1[2]], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:2]
        v4 = torch.cat([x1[0], x1[2], x1[1]], dim=1)
        return v4
   
# Initializing the model
m = Model()

# Inputs to the model
x1 = [torch.randn(3, 3, 64, 64), torch.randn(3, 5, 64, 64), torch.randn(3, 1, 64, 64)]
