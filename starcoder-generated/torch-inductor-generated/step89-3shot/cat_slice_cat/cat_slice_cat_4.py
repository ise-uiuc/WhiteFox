
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x5):
        v1 = torch.cat([x1, x5], dim=1)
        v4 = torch.cat([v1[:, 0:9223372036854775807], v1[:, 0:x1.size()[1]]], dim=1)
        return v4

# Initializing the model
x1 = torch.randn(10, 10)
x5 = torch.randn(10, 10)
m = Model()
