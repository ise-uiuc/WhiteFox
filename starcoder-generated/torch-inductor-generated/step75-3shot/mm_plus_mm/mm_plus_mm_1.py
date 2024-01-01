
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        p1 = torch.mm(x1, x2)
        p2 = torch.mm(x2, x2) # This is the same as torch.mm(x2, x1), since mats are square
        p3 = torch.mm(p2, x2)
        p4 = torch.mm(x2, x2)
        p5 = p4 + p4
        return (p1 + p2) * p3 + p5
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
