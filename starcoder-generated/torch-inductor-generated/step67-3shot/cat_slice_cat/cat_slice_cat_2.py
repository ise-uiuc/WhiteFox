
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x2 = torch.cat(x, dim=1)
        x3 = x2[:, int(3) : int(9223372036854775807)]
        x4 = x3[:, 0:size]
        x5 = torch.cat([x2, x4], dim=1)            
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x = [torch.randn(1, 6, 64, 64), torch.randn(1, 6, 64, 64), torch.randn(1, 6, 64, 64)]
