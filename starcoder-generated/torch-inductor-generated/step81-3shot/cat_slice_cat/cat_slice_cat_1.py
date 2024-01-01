
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat(x1, x2, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:-1]
        v4 = [v1, v3]
        v5 = torch.cat(v4, dim=1)
        return v5
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 224, 224)
x2 = torch.randn(1, 7000000000, 224, 224)
