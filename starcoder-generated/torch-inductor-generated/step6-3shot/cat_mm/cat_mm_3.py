
class Model(torch.nn.Module):
    def __init__(self, a3):
        super().__init__()
        self.a3 = a3
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        if self.a3 == 1:
            return torch.cat([v2, v2, v2, v2, v2], 1) + torch.cat([v1, v1, v1, v1, v1, v1, v1, v1], 1)
        if self.a3 == 2:
            return torch.cat([v1, v1, v1, v1], 1) + torch.cat([v2, v2, v2, v2, v2, v2, v2, v2], 1)
        if self.a3 == 3:
            return torch.cat([v2, v2, v2, v2, v2, v2], 1)
        if self.a3 == 4:
            return torch.cat([v1, v1, v1, v1, v1, v1, v1, v1], 1)
        if self.a3 == 5:
            return torch.cat([v2, v2, v2, v2], 1)
        if self.a3 == 6:
            return torch.cat([v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
a3 = 1 # change a3's value to {2, 3, 4, 5, 6} to generate different results
