
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 20)
        self.linear2 = torch.nn.Linear(20, 20)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

...

