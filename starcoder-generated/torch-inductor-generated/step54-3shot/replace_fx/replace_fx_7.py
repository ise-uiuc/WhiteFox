
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 12)
        self.bn1 = torch.nn.BatchNorm1d(12)
        self.dropout1 = torch.nn.Dropout(p=0.05)
        self.linear2 = torch.nn.Linear(12, 12)
        self.bn2 = torch.nn.BatchNorm1d(12)
        self.dropout2 = torch.nn.Dropout(p=0.02)
        self.activation = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(12, 8)
        self.bn3 = torch.nn.BatchNorm1d(8)
        self.dropout3 = torch.nn.Dropout(p=0.15)
        self.linear4 = torch.nn.Linear(8, 1)
    def forward(self, x):
        r0 = self.dropout1(x)
        r1 = self.linear1(r0)
        r2 = self.bn1(r1)
        r3 = self.activation(r2)
        r4 = self.dropout2(r3)
        r5 = self.linear2(r4)
        r6 = self.bn2(r5)
        r7 = self.activation(r6)
        r8 = self.dropout3(r7)
        r9 = self.linear3(r8)
        r10 = self.bn3(r9)
        r11 = self.activation(r10)
        r12 = self.linear4(r11)
        return r12        
# Inputs to the model
x1 = torch.randn(1, 5)
