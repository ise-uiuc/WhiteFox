
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(2, 2)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dropout3 = torch.nn.Dropout(0.3)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.dense1(x1)
        x3 = self.relu1(x2)
        x4 = self.dropout1(x3)
        x5 = self.dropout2(x3)
        x6 = self.dropout3(x3)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
