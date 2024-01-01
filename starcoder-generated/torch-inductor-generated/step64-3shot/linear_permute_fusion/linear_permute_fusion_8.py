
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.linear_1(x1))
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.relu(self.linear_2(x1))
        v4 = v3.permute(0, 2, 1)
        v5 = torch.nn.functional.relu(self.linear_1(x1))
        v6 = v5.permute(0, 2, 1)
        return v6
# Inputs to the model
x1 = torch.randn(3, 2, 2)
