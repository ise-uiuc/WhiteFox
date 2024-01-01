
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(0.0)
        self.dropout2 = torch.nn.Dropout(0.1)
    def forward(self, x1):
        a = torch.randint(2, (2, 2)) 
        b = self.dropout1(a)
        c = self.dropout1(b)
        d = self.dropout2(c)
        e = self.dropout2(d)
        return e
# Inputs to the model
x1 = torch.randn(1, 2, 2) 
