
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = self.fc1(x1)
        x3 = self.fc2(x1)
        t1 = torch.rand_like(x2)
        t2 = torch.rand_like(x3)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(2, 2)
