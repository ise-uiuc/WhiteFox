
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 100)
        self.fc2 = torch.nn.Linear(100, 0.2)
    def forward(self, x0):
        v1 = self.fc1(x0)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10 * self.fc2(v10)
# Inputs to the model
x0 = torch.randn(1, 100)
