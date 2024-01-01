
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x1):
       #print(x1.shape)
       v1 = self.linear(x1)
       v2 = v1 * torch.FloatTensor([0.5]).to(device)
       v2 = torch.reshape(v2,(v2.size()[0],32))
       v3 = (self.linear(x1)*(self.linear(x1)*torch.FloatTensor([3.0]).to(device)))*torch.FloatTensor([0.044715]).to(device)
       v4 = v3 * torch.FloatTensor([0.7978845608028654]).to(device)
       v5 = torch.tanh(v4)
       v6 = v5 + 1
       v7 = v2 * v6
       #print("v7")
       #print(v7.shape)
       #print("v2")
       #print(v2.shape)
       #print("v6")
       #print(v6.shape)
       return v7


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
