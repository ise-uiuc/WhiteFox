
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(64, 2)
    def forward(self, x1):
        w = x1.view((2, 4, 16))
        v1 = self.l0(w)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.sum(v2, dim=1, keepdim=True)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64)
