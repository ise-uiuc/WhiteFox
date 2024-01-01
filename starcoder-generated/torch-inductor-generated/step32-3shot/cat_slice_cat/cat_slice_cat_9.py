
class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x1, x2, x3):
        x40, 10 = torch.split(x1, 10)
        x50, 10 = torch.split(x2, 10)
        x60, 10 = torch.split(x3, 10)  
        x41, x51, x61 = torch.split(x40, 32), torch.split(x50, 32), torch.split(x60, 32)    
        res1 = torch.cat([x41,x51,x61], dim=1)  
        res2, _  = torch.split(res1, 96)  
        res2, _  = torch.split(res2, 96)  
        res3, _  = torch.split(res2, 96)  
        return res3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 32)
x2 = torch.randn(1, 100, 32)
x3 = torch.randn(1, 100, 32)
