
class Model(torch.nn.Module):
    def __init__(self, m1, m2):
        super(Model, self).__init__()
        self.m1 = m1
        self.m2 = m2
 
    def forward(self, x):
        v1 = self.m1(x)
        v2 = self.m2(x)
        print('Model:', v1.detach(), v2.detach())
        return v1 * 0.5 + v2 + 1

# Initializing the model
model = Model(nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([5.0])))
  
# Inputs to the model
x = torch.randn(2, 3)
model(x)

# Replacing modules with a custom module
model.m1 = nn.Parameter(torch.tensor([32.0]))
model(x)
model.m1 = nn.Parameter(torch.tensor([64.0]))
model(x)
model.m1 = nn.Parameter(torch.tensor([128.0]))
model(x)

# Replacing modules with a custom module class
class CustomOp(torch.nn.Module):
    def __init__(self):
        super(CustomOp, self).__init__()
 
    def forward(self, x):
        return x + x + x

model.m1 = CustomOp()
model(x)
model.m1 = CustomOp()
model(x)


