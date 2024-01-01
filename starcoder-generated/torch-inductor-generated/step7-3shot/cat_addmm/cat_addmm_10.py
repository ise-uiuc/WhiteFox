
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.addmm = torch.nn.functional.linear
        self.cat = torch.cat
 
    def forward(self, x2):
        v1 = self.addmm(x2, torch.randn(64, 2560), torch.randn(2560, 480))
        v2 = self.cat([v1], dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(480)
