
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(10, 16)
        self.dense2 = torch.nn.Linear(16, 5)
        self.dense3 = torch.nn.Linear(5, 256)
        self.dense4 = torch.nn.Linear(256, 1024)
    def forward(self, x1, x2):
        v1 = torch.sigmoid(self.dense1(x1))
        v2 = v1 + x1
        v3 = self.dense2(v2)
        v4 = v3 + self.dense3(x2)
        v5 = torch.sin(v4)
        return self.dense4(v5)
# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 256)
