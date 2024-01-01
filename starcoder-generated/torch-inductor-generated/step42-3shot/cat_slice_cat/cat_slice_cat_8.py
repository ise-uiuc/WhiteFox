
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat1 = torch.nn.Linear(123456789, 123456789)
        self.cat2 = torch.nn.Linear(654321, 654321)
        self.cat3 = torch.nn.Linear(876543, 876543)
 
    def forward(self, x1, x2, x3, x4):
        l1 = self.cat1(x1)
        l2 = self.cat2(x2)
        l3 = self.cat3(x3)
        v1 = torch.cat([l1, l2, l3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 123456789)
x2 = torch.randn(1, 654321)
x3 = torch.randn(1, 876543)
x4 = torch.randn(1, 123456789)
