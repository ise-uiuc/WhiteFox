
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        c1 = torch.cat([x1, x2], dim=1)
        c2 = c1[:, :64]
        s1 = c2[:, 0:32767]
        c3 = torch.cat([c1, s1], dim=1)
        return c3

# Initializing the model
m = Model()

# Outputs of each layer
