
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(20, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
        )
 
    def forward(self, x1):
        v1 = self.linears(x1)
        v2 = torch.cat((v1, torch.randn(1, 10)), dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
