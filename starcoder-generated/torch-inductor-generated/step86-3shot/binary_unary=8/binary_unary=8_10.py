
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.module3 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.module2 = torch.nn.ReLU()
    def forward(self, x1):
        v10 = x1.clone()
        v1 = self.module1(x1)
        v11 = v10 + v1
        v2 = self.module3(x1)
        v12 = v11 + v2
        v3 = self.module2(v12)
        v4 = torch.cat([v1, v2, v3], 1) # Concatenate along channel axis.
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
