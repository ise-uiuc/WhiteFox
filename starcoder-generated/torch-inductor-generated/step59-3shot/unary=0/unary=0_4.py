
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, stride=1, padding=0)
        self.fc = torch.nn.Linear(1000, 111)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = v10.reshape(v10.size()[0], 1000)
        v12 = self.fc(v11)
        return v1 + v12
# Inputs to the model
x3 = torch.randn(5, 1, 223, 250)
