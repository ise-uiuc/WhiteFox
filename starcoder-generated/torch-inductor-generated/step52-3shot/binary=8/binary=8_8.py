
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Linear(1, 1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.softmax(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3)
