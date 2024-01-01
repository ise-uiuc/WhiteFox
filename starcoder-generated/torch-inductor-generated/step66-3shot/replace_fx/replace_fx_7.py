
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = F.dropout(x2)
        x4 = F.relu(self.fc(x3))
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
