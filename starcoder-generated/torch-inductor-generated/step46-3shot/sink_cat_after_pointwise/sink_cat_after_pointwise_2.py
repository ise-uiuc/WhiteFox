
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 2, stride=1),
            torch.nn.Linear(3*10*10, 4)
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 2),
            torch.nn.Linear(15*5*5, 4)
        )
    def forward(self, x):
        a = self.branch1(x)
        b = self.branch2(x)
        return torch.cat((a, torch.relu(b), torch.relu(a), b, torch.relu(a)), dim=1)
# Inputs to the model
x = torch.randn(1, 2, 3, 3)
