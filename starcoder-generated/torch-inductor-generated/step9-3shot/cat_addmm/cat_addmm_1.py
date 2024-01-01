
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 1, bias=True)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2)
        v2 = torch.cat([v1.unsqueeze(0), v1.unsqueeze(1)], dim=0).view(2, -1)
        v3 = self.fc(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 4)
x2 = torch.randn(4, 2)
