
class Model(torch.nn.Module):
    def forward(self,x,y,z):
        t1 = torch.mm(x,y)
        t2 = torch.mm(z,x)
        return t1 + t2
# Inputs to the model
x = torch.randn(512,512)
y = torch.randn(512,512)
z = torch.randn(512,512)
