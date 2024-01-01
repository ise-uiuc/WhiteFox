
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v = self.linear(x)
        v = self.relu(v)
        v = v - 1
        return v
# Inputs to the model
x = torch.rand(1, 2)
