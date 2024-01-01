
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x0):
        x1 = F.dropout(x0)
        x2 = F.dropout(x1, p=0.5)
        x3 = F.dropout(x2, p=0.5)
        x4 = F.dropout(x3, p=0.5)
        return x4 # The last 3 dropout calls are folded into one
