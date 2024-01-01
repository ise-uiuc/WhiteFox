
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
        self.dropout = torch.nn.Dropout(p1)
    def forward(self, x1):
        x = self.dropout(x1)
        t = F.relu(x)
        x2 = F.relu(x)
        x3 = F.relu(x)
        return x3
p1 = 0.3
# Inputs to the model
x1 = torch.randn(2, 2, 2)
