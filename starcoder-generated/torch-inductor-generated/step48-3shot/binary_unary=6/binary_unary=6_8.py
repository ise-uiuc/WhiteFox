
class Model(torch.nn.Module):
    def __init__(self, n):
        if n<0 or not isinstance(n,int):
            raise ValueError("'n' must be a non-negative integer")
        self.n = math.ceil(n)
        super().__init__()
     
    def forward(self, x1):
        x = list()
        out = torch.ones_like(x1[:,0,:,:])*x1[:,0,:,:].mean()
        x.append(out)
        for i in range(0,self.n):
            out = torch.cos(x1.mean())
            x.append(out)
        return x

# Initializing the model
m = Model(5)

# Inputs to the model
x1 = torch.randn(6, 1, 100, 100)
