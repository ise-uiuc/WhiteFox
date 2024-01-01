
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([[-2.0]])
x2 = torch.tensor([[-1.5]])
x3 = torch.tensor([[-1.0]])
x4 = torch.tensor([[0.0]])
x5 = torch.tensor([[0.5]])
x6 = torch.tensor([[1.0]])
x7 = torch.tensor([[10.0]])
