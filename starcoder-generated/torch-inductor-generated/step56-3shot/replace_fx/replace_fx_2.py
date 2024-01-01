
class Model1(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(1, out_features)
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x
# Inputs to the model
x1 = torch.randn(1, 1)
