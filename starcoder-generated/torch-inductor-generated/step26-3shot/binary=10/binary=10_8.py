
def f1(x):
    return 2 * x

def f2(x, y):
    return x + 2 * y

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(8, 16)
        self.linear1 = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 64)
        self.linear3 = torch.nn.Linear(64, 128)
        self.linear4 = torch.nn.Linear(128, 256)
        self.linear5 = torch.nn.Linear(256, 512)

    def forward(self, x):
        a1 = self.linear0(x)
        a2 = a1 + f1(a1)
        a3 = a2 + f1(a2)
        a4 = a3 + f1(a3)
        a5 = a4 + f1(a4)

        b1 = self.linear1(a5)
        b2 = b1 + f2(b1, b1)

        c1 = self.linear2(b2)
        c2 = c1 + f2(c1, c1)

        d1 = self.linear3(c2)
        d2 = d1 + f2(d1, d1)

        e1 = self.linear4(d2)
        e2 = e1 + f2(e1, e1)

        f1 = self.linear5(e2)
        f2 = f1 + f2(f1, f1)

# Initializing the model
m = Model()

random_input = torch.randn(8)
