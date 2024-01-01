
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        t1 = torch.mm(x1, x2) # Matrix multiplication between x1 and x2
        t2 = torch.mm(x3, x1) # Matrix multiplication between x3 and x1
        out = torch.mm(x2, x3) # Matrix multiplication between x2 and x3
        return out + out + t1
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
