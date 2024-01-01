
class Model(torch.nn.Module):
    # TODO: Write your solution here
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(21, 21)
        self.linear_2 = torch.nn.Linear(21, 5)
        self.linear_3 = torch.nn.Linear(5, 16)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = torch.relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 70)
