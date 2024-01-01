
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v2 = x1.permute(0,2,1)
        v1 = self.linear1(v2)
        v1 = torch.nn.functional.relu(v1)
        v1 = self.linear2(v1)
        v3 = v1 * 2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
