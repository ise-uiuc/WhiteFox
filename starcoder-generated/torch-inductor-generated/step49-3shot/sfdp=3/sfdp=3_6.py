
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(3, 32)
        self.k = torch.nn.Linear(32, 32)
        self.q = torch.nn.Linear(32, 32)
        self.dropout1d = torch.nn.Dropout(0.2)
        self.scale_factor = np.sqrt(32)
 
    def forward(self, x1, x2, x3):
        y1 = self.w(x1)
        y2 = self.w(x2)
        y3 = self.w(x3)
        y4 = self.k(y1)
        y5 = self.k(y2)
        y6 = self.k(y3)
        y7 = self.q(y1)
        y8 = self.q(y2)
        y9 = self.q(y3)
        y10 = torch.matmul(y7, y5.transpose(-2, -1))
        y11 = torch.matmul(y8, y6.transpose(-2, -1))
        y12 = torch.matmul(y9, y4.transpose(-2, -1))
        y13 = y10 * self.scale_factor
        y14 = y11 * self.scale_factor
        y15 = y12 * self.scale_factor
        z1 = y13.softmax(dim=-1)
        z2 = y14.softmax(dim=-1)
        z3 = y15.softmax(dim=-1)
        z4 = self.dropout1d(z1)
        z5 = self.dropout1d(z2)
        z6 = self.dropout1d(z3)
        return (
            z4.matmul(y4) + z5.matmul(y5) + z6.matmul(y6),
            z4, y4,
            z5, y5,
            z6, y6,
        )

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
x3 = torch.randn(1, 3)
__output__, q, k, p1, k1, p2, k2 = m(x1, x2, x3)

