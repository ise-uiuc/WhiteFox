
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.dropout3 = torch.nn.Dropout(p=0.3) 
    def forward(self, x1):
        a1 = self.dropout1(x1)
        a2 = self.dropout2(x1)
        a3 = self.dropout3(x1)
        return torch.abs(a3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
