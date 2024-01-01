
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear_1 = torch.nn.Linear(2, 2)
        # self.linear_2 = torch.nn.Linear(2, 2)
        # self.linear_3 = torch.nn.Linear(2, 2)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
        )
    def forward(self, x1):
        v1 = self.linear(x1)
        return x1 - v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
