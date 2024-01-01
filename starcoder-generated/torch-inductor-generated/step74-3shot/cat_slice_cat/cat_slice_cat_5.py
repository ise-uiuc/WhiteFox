
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        r1 = [x1]
        r2 = torch.cat(r1)
        r3 = r2[:, 0:x2]
        r4 = torch.cat(r1, r3)
        return r4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randint(1, x1.shape[1], (1,))
x3 = torch.randint(1, x1.shape[1], (1,))

