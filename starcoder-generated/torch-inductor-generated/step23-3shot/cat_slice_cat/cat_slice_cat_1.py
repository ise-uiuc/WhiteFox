
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t1, t2, t3, size=9223372036854775807):
        t4 = torch.cat([t1, t2], dim=1)
        t5 = t1[:, 0:9223372036854775807]
        t6 = t4[:, 0:size]
        v1 = torch.cat([t4, t6], dim=1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(1, 3, 64, 64)
t2 = torch.randn(1, 3, 64, 64)
t3 = torch.randn(1, 3, 64, 64)
