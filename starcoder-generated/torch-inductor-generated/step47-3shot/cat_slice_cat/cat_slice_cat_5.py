
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x4):
        v1 = torch.cat(x4, dim=1)
        v2 = v1[:, -1]
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x4_1 = torch.randn(1, 7680)
x4_2 = torch.randn(1, 1600000)
x4_3 = torch.randn(1, 10000000)
x4 = [x4_1, x4_2, x4_3]
