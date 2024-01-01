
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = self.linear1(x1).permute(0, 2, 1)
        v2 = v1.float().sum(dim=2, keepdim=True)
        v3 = self.linear2(v2)
        return v3.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
