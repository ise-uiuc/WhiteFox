
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(2, 2)
        self.dense2 = torch.nn.Linear(2,2)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dropout3 = torch.nn.Dropout(0.3)
    def forward(self, batch):
        x1 = self.dense1(batch)
        x2 = self.dense2(x1)
        x3 = self.dropout1(x1)
        x4 = self.dropout2(x2)
        x5 = self.dropout3(x2)
        return x2
# Inputs to the model
batch = torch.randn(1, 10)
