
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        c = torch.cat([x1, x2], dim=1)
        s = c[:, 0:9223372036854775807]
        r = c[:, 0:int(x1.shape[1])]
        a = torch.cat([c, s], dim=1)
        return a
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
