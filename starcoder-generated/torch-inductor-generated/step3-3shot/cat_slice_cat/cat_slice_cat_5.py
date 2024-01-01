
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = [2, 3, 4, 5]
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        if self.shape[3]!= v2.shape[1]:
            x3 = torch.full((v1.shape), 12.0, dtype=torch.float32)
            v3 = torch.cat([v1, x3], dim=1)
            v4 = v3[:, 0:self.shape[3]]
        else:
            v4 = v2[:, 0:self.shape[3]]
        v5 = v4 + 1
        return v5


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
x2 = torch.randn(1, 3, 2, 5)
m.shape = x2.shape[2:4]
