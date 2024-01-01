
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = torch.nn.Linear(64*64, 64)
        self.fc_2 = torch.nn.Linear(64, 10)
 
    def forward(self, x1):
        x2 = x1.view(-1, 64*64)
        v1 = self.fc_1(x2)
        v2 = torch.addmm(v1, v1.t(), x2)
        v3 = torch.cat([v2], dim=1)
        v4 = self.fc_2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
