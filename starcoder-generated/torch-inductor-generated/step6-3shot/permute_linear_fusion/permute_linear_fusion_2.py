
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(x1)
        v2 = torch.nn.functional.sigmoid(x1)
        return v2
# Inputs to the model
x1 = torch.randn(1,)
