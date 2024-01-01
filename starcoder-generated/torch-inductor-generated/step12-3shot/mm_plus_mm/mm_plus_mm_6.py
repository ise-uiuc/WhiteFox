
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(torch.ones((2, 2)),torch.ones((2, 2)))
        t2 = torch.mm(torch.ones((2, 2)), torch.ones((2, 2)))
        t3 = t1 + t2
        return t3
# Inputs to the model
input = torch.randn(2, 2)
