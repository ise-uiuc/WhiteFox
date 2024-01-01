
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.5)
        t2 = torch.rand_like(t1)
        t3 = F.avg_pool2d(t2, 3)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
