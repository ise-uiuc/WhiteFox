
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
    def forward(self, x):
        x = torch.rand_like(x)
        x = self.dropout1(x)
        return x
x = torch.randn(3)
