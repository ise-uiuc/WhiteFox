
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(0.8739)
        self.dropout2 = torch.nn.Dropout(0.2876)
        self.linear1 = torch.nn.Linear(75, 72)
        self.linear2 = torch.nn.Linear(72, 44)
        self.linear3 = torch.nn.Linear(44, 87)
    def forward(self, x1):
        v1 = self.dropout1(x1)
        v2 = self.linear1(v1)
        v3 = torch.relu(v2)
        v4 = self.dropout2(v3)
        v5 = self.linear2(v4)
        v6 = torch.relu(v5)
        v7 = self.linear3(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(4, 75)
