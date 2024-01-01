
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        h1 = torch.mm(x1, x2)+torch.mm(x1, x3)+torch.mm(x2, x3)
        h2 = torch.mm(x1, x1)+torch.mm(x2, x1)+torch.mm(x3, x1)
        h3 = torch.mm(x1, x1)+torch.mm(x2, x2)+torch.mm(x3, x3)
        return h1 + h2 + h3
# Inputs to the model
x1 = torch.randn(50, 50) 
x2 = torch.randn(50, 50) 
x3 = torch.randn(50, 50) 
