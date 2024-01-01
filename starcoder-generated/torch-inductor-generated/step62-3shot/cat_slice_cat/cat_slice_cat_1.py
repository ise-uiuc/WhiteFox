
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        return torch.cat([x1[:, 0:9223372036854775807, 0:size], x1], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 2, 100000)
