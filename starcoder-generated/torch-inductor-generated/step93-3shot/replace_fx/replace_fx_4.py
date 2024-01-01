
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = F.dropout(x1, p=0.5, training=self.training)
        return t1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5, training=True)
        t1 = F.dropout(x2, p=0.5, training=self.training)
        return t1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
