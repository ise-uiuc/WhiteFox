
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = torch.nn.Linear(2, 2)
        self.linear = torch.nn.Linear(2, 1)
 
    def forward(self, x1):
        x = torch.transpose(x1,2,3) #.to("cuda")
        x = x.reshape(-1,300)
        x = self.linear(x)
        v2 = x * 0.5
        v4 = torch.erf(x * 0.7071067811865476)
        v5 = v4 + 1
        y = v2 * v5
        y = torch.transpose(y.reshape(1,4,-1,300),2,3)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,4,300,2)
