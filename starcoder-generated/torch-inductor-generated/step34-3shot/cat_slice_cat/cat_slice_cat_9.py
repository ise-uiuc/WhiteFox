
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *args):
        v1 = [arg for arg in args][0]
        v2 = v1[:, :, :, 0:size]
        v4 = [v1, v2]
        v5 = torch.cat(v4, dim=3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
x2 = torch.randn(1, 3, 20, 20)
x3 = torch.randn(1, 3, 30, 30)
