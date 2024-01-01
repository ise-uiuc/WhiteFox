
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(7,1024)
        self.dense2 = torch.nn.Linear(1024,128)
        self.dense3 = torch.nn.Linear(128,512)
        self.dense4 = torch.nn.Linear(512,4)
    def forward(self, x1):
        v1 = self.dense1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = self.dense2(v3)
        v5 = v4 - 100
        v6 = F.relu(v5)
        v7 = self.dense3(v6)
        v8 = v7 - 1000
        v9 = F.relu(v8)
        v10 = self.dense4(v9)
        v11 = v10 - 1000
        v12 = F.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 7)
