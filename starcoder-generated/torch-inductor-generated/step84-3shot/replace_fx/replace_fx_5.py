
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = torch.nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.d(x)
        x = F.dropout(x, p=0.5)
        return x
