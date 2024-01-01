
class Model(torch.nn.Module):
    def __init__(self, n, d, h, s, p):
        super().__init__()
        self.dense1 = torch.nn.Linear(n, d)
        self.dense2 = torch.nn.Linear(d, h)
        self.dense3 = torch.nn.Linear(h, n)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p)
 
    def forward(self, x):
        v1 = self.dense1(x)
        v2 = self.dense2(v1)
        v3 = self.dense3(v2)
        v4 = self.dropout(self.softmax(v3 * 10))
        v5 = v3.mm(v4.transpose(0, 1))
        return v5


# Initializing the model
m = Model(n=5, d=1, h=3, s=1, p=0.5)

# Inputs to the model
x1 = torch.randn(4, 5)
