
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t1, t2):
        t3 = torch.cat([t1, t2], dim=1)
        t4 = t3[:, 0:9223372036854775807]
        t5 = t4[:, 0:5052907553014525696]
        t6 = torch.cat([t3, t5], dim=1)
        return t6

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(1, 10, 256, 256)
t2 = torch.randn(1, 33, 256, 256)
