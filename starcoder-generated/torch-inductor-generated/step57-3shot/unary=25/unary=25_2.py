
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(10, 16)

    def forward(self, x):
        v1 = self.dropout(x)
        v2 = self.linear(v1)
        v3 = v2 > 0
        v4 = v2 * 0.1
        v5 = torch.where(v3, v2, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
