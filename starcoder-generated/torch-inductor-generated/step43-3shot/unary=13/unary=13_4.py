
class Model(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Model, self).__init__()
        self.i2h = torch.nn.Linear(in_dim, hidden_dim)
        self.h2o = torch.nn.Linear(hidden_dim, out_dim)
        self.s2h = torch.nn.Linear(in_dim, hidden_dim)
        self.s2o = torch.nn.Linear(hidden_dim, out_dim)
        self.h = None

    def forward(self, x):
        h_sig = torch.sigmoid(self.i2h(x) + self.h2o(self.s2h(x)))
        self.h = h_sig * self.h if self.h is not None else h_sig
        o = self.h + self.h2o(self.s2o(x))

        return o

# Initializing the model
random.seed(0)
m = Model(784, 64, 10)

# Input to the model
x2 = torch.randn(1, 784)

