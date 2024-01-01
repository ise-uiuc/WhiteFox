
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant1 = torch.tensor([1,2,3,4,5,6,7,8,9])
        self.constant2 = torch.tensor([9,8,7,6,5,4,3,2,1])

    def forward(self, x1, x2):
        v1 = torch.cat([x1,x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = x2[:, 0:6]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 10)
x2 = torch.randn(1, 1, 6)
