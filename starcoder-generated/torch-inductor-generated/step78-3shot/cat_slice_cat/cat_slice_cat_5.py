
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
 
    def forward(self, x1):
        x2 = torch.transpose(x1,0,2)
        x3 = torch.transpose(x2,0,1)
        v1 = torch.cat([x1,x3])
        v2 = v1[:,0:9223372036854775808]
        v3 = v2[:,0:64]
        v4 = torch.cat([v1,v3],1)
        v5 = torch.transpose(v4,0,2)
        v6 = torch.transpose(v5,0,1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,1,9,9)
