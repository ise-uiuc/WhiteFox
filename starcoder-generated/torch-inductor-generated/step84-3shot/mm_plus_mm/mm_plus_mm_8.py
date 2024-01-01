
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        a = (torch.mm(x, z*z)+1)*2 + torch.mm(x, y + y)
        b = ((torch.mm(y, z)+2)*3)*4 + torch.mm(y, z*z*z) 
        c = (torch.mm(z, x)+3)*2 + torch.mm(z, y+z*y*y*z*x)
        return a/b+c
# Inputs to the model
x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn(5, 5)
