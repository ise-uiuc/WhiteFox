
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        r = torch.cat([x, y], dim=1)
        output = r[:, : -int(r.shape[1] / 2) or None]
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 5, 6, 8)
y = torch.randn(2, 3, 6, 8)
