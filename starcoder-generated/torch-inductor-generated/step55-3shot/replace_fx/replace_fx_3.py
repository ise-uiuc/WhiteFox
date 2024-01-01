
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.DropOut(p=0.2)
        self.dropout1 = F.dropout

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout1(x, p=0.2)
        return x
# Inputs to the model
x1 = torch.randn(2, 3, 4)
