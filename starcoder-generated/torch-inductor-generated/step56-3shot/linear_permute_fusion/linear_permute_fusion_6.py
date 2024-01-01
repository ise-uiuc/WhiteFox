
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v4 = torch.nn.functional.relu(x1)
        v2 = v4
        v8 = torch.nn.functional.relu(self.linear_2(self.linear_1(v2)))
        v12 = v8.permute(2, 0, 1)
        return torch.nn.functional.relu(v12)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
