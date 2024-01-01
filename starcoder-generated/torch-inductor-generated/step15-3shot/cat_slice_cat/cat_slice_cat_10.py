
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.concat = torch.cat
    
    def forward(self, x1, x2):
        t1 = self.concat([x1, x2], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:64]
        t4 = self.concat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 4, 4)
x2 = torch.randn(1, 64, 8, 8)
