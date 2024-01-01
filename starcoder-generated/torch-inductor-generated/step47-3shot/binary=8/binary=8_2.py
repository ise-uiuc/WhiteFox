
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(192, 1)
        self.gelu = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.6539580252647174)
        self.fc2 = torch.nn.Linear(1, 1)
        self.gelu1 = torch.nn.GELU()
        self.drop1 = torch.nn.Dropout(0.6883371296619819)
        self.fc3 = torch.nn.Linear(1, 1)
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = self.gelu(v1)
        v3 = self.drop(v2)
        v4 = self.fc2(v3)
        v5 = self.gelu1(v4)
        v6 = self.drop1(v5)
        v7 = self.fc3(v6)
        return v7
# Inputs to the model
x = torch.randn(1, 192)
