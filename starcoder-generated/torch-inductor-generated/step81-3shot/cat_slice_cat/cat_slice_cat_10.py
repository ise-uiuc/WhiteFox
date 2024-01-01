
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2, x3):
        v1 = torch.cat((x2, x3))
        v4 = torch.cat((v1[:, 0:9223372036854775807], v1[:, 0:v1.shape[1]]))

        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 65, 64)
