
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()
        self.linear1 = torch.nn.Linear(6400, 16777216)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1[0]
        v2 = self.softmax(v1)
        v3 = self.linear1(v2)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6400)
