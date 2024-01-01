
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(5, 64)
    def forward(self, x1):
        v1 = self.dense(x1)
        v2 = v1
        v3 = v1
        v4 = v2
        v5 = v3
        v6 = v4
        v7 = v5
        v8 = v6
        v9 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 5)
