
class Model(torch.nn.Module):
    def __init__(self, size):
        super(Model, self).__init__()
        self.size = size

    def forward(self, x1, x2):
        s1 = torch.cat([x1, x2], dim=1)
        s2 = s1[:, :self.size]
        s3 = s1[:, self.size:]
        s4 = torch.cat([s1, s3], dim=1)
        return s4

# Initializing the model
m = Model(5)

# Input tensors to the model
x1 = torch.randn(1, 9, 3, 3)
x2 = torch.randn(1, 4, 3, 3)
