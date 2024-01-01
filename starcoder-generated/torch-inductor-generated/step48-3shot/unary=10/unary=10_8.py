
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
        self.relu6 = torch.nn.ReLU6(6, 1)

    def forward(self, x1):
        l1 = self.fc(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Input to the model
input = torch.randn(1, 3)
__output = m(input)

