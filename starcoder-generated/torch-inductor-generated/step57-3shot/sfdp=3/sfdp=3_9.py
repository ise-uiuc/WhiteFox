
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(10, 128)
        self.m2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        q = self.m1(x)
        k = self.m1(x)
        dots = torch.matmul(q, k.t())
        scale = math.sqrt(dots.size(-1))
        softmax = dots.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax, 0.5)
        v = self.m2(dropout)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.rand(51, 10)
