
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(1, 3, kernel_size=2)
        self.l2 = torch.nn.Conv2d(1, 2, kernel_size=2)
        self.l3 = torch.nn.Conv2d(3, 1, kernel_size=2)
        self.l4 = torch.nn.Sequential(
            #...
        )
        self.l5 = torch.nn.Sequential(
            #...
        )
        self.l6 = torch.nn.Linear(10, 10)
    def forward(self, x1, x2):
        t1 = F.relu(self.l2(F.relu(self.l1(x1))))
        t2 = F.relu(self.l4(x1))
        t3 = F.relu(self.l5(t2))
        t4 = torch.add(t1, t3)
        t5 = self.l3(t4)
        t6 = F.relu(t5)
        t7 = F.relu(t6)
        return torch.cat((self.l6(t4), F.relu(self.l6(t7))), dim=0)
# Inputs to the model
x1 = torch.randn(1, 1, 3, 2)
x2 = torch.randn(1, 1, 3, 2)
