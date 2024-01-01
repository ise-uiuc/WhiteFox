
class Model(torch.nn.Module):
    def __init__(self, x1):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.8)
        self.fc = torch.nn.Linear(in_features=x1.size(1), out_features=8, bias=True)
    def forward(self, x1):
        v1 = self.dropout(x1)
        v2 = torch.relu(v1)
        return torch.relu(self.fc(v2))
# Inputs to the model
x1 = torch.randn(1, 512, 100, 100)
