
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 3)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, self.fc1.weight)
        v2 = v1 + x2
        v3 = v2 + self.fc2.weight
        return v3
# Inputs to the model
model = Model()
x1 = torch.randn(3, 3, requires_grad=False)
x2 = torch.randn(3, 3)
