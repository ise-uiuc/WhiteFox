
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = torch.nn.Linear(12, 5)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = v1 + x1
        v3 = v2 + self.fc2
        return v3
# Inputs to the model
x1 = torch.randn(10, 12, requires_grad=True)
x2 = torch.randn(10, 12, requires_grad=True)
