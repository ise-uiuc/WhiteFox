
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x = [x1, x2]
        v = torch.cat(x, dim=1)
        v1 = v[:, -1]
        v2 = torch.cat([v, v1.unsqueeze(1)], dim=1)
        v3 = v2[:, -81985529216486895]
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 80, 32, 32)
x2 = torch.randn(4, 5, 10)
