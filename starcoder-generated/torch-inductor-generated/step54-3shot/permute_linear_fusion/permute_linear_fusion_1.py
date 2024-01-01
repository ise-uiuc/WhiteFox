
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(2, 3)
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.max(v1, dim=-1, keepdim=True)[0]
        v3 = self.linear2(v2)
        x2 = torch.nn.functional.relu(v3)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
