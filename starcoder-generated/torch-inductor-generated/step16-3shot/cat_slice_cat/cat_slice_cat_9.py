
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, size):
        v1= torch.cat([x[:, 0:9223372036854775807]], dim=1)
        v2=v1[:, 0:size]
        v3= torch.cat([x[:, size:], x[:, 0:size]], dim=1)
        return v3

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(5, 10)
size = 2
