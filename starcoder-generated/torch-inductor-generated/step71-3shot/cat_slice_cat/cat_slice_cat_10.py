
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        x = torch.cat([x1, x2], dim=1)
        v1 = x[:, 0:]
        v2 = x[:, 9223372036854775807:9223372036854775808]
        v3 = x[:, 0:size]
        v4 = torch.cat([x, v3], dim=1)    
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 111, 28)
x2 = torch.randn(1, 567, 27)
x3 = torch.randn(1, 322, 27)
x4 = torch.randn(1, 388, 27)
x5 = torch.randn(1, 283, 27)
