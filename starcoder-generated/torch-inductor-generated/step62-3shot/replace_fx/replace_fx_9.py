
class Model(torch.nn.Module):
    def __init__0(self):
        super().__init__()
    def forward__0(self, _x):
        x = torch.rand_like(_x)
        x2 = torch.randn(_x.size(0),_x.size(2),_x.size(3))
        x3 = torch.rand(1,_x.size(3),dtype=_x.dtype,device=_x.device)
        x4 = torch.mul(x3,x)
        x5 = torch.mul(x4,x)
        x6 = torch.mul(x5,x)
        x7 = torch.mul(x6,x)
        x8 = torch.mul(x7,x)
        x9 = torch.mul(x8,x)
        x10 = torch.mul(x9,x)
        x11 = torch.mul(x10,x)
        x12 = torch.mul(x11,x)
        x13 = torch.mul(x12,x)
        x14 = torch.mul(x13,x)
        x15 = torch.mul(x14,x)
        x16 = torch.mul(x15,x)
        x17 = torch.mul(x16,x)
        x18 = torch.mul(x17,x)
        x19 = torch.mul(x18,x)
        x20 = torch.div(x19,x)
        x21 = torch.add(x2,x20)
        x22 = torch.sub(x21,v1)
        return x22
# Inputs to the model
self.x = torch.randn(37, 31)
