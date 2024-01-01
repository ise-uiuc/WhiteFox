
class MyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        X = self.conv(X)
        X = X * 0.7071067811865476
        X = torch.nn.functional.dropout(X, p=0.4)
        return X
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 128, 3, stride=2, padding=1)
    def forward(self, X):
        X1 = MyConv()(X)
        return X1
# Inputs to the model
X = torch.randn(1, 12, 32, 32)
