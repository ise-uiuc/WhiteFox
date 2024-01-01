
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 5)
        self.linear3 = torch.nn.Linear(5, 20)
        self.linear4 = torch.nn.Linear(20, 10)
        self.linear5 = torch.nn.Linear(10, 7)
        self.linear6 = torch.nn.Linear(7, 2)
    def forward(self, x):
        f = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(f, p=0.5)
        x = self.linear1(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=False)
        x1 = torch.nn.functional.relu(x)
        x2 = torch.rand_like(x)
        x3 = self.linear2(x1)
        x4 = torch.nn.functional.dropout(x3, p=0.3)
        x5 = torch.rand_like(x3)
        x6 = self.linear3(x4)
        x7 = torch.nn.functional.dropout(x6, p=0.3)
        z1 = torch.relu(x7)
        z11 = torch.rand_like(z1)
        z2 = self.linear4(z1)
        z22 = torch.rand_like(z2)
        z3 = self.linear5(z2)
        z33 = torch.rand_like(z3)
        z4 = self.linear6(z3)
        z44 = torch.rand_like(z4)
        y1 = z4
        y2 = z44
        return x
# Inputs to the model
x1 = torch.randn(1, 3)
