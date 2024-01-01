
class Model(torch.nn.Module):
    def forward(self,x1,x2,x3,x4):
        result = torch.mm(x2,x1)
        result = torch.mm(x3,x1)
        result = torch.mm(x4,x1)
        result = torch.mm(x4,x1)
        result = torch.mm(x2,x1)
        result = torch.mm(x2,x1)
        result = torch.mm(x3,x1)
        result = torch.mm(x4,x1)
        result = torch.mm(x4,x1)
        result = torch.mm(x2,x1)
        for i in range(10):
            result = torch.mm(x2,x1)
        return result
# Inputs to the model
x1 = torch.randn(3,3)
x2 = torch.randn(3,3)
x3 = torch.randn(3,3)
x4 = torch.randn(3,3)
