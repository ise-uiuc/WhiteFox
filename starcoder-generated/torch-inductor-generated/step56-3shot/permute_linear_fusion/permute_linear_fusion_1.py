
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 20)
        self.linear2 = torch.nn.Linear(20, 20)
        self.linear3 = torch.nn.Linear(20, 2)
    def forward(self, x1):
        x1 = x1.detach()
        v1 = self.linear1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.linear2(v2)
        v4 = torch.nn.functional.relu(v3)
        v5 = self.linear3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
