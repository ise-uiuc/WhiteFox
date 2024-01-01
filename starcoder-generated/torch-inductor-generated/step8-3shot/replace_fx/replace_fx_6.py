
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(in_features=3, out_features=4)
        self.dropout = torch.nn.Dropout(p=0.5)  # 0.5, 0.33
        self.b = torch.nn.Linear(in_features=4, out_features=5)
        self.c = torch.nn.Sequential(
            torch.nn.Linear(in_features=5, out_features=3),
            torch.nn.Sigmoid(),
        )
    def forward(self, x1):
        a = torch.reshape(x1, (1, 3))
        b = self.dropout(a)
        c = self.a(b)
        d = self.b(c)
        e = self.c(d)
        return e
# Inputs to the model
x1 = torch.randn(3)
