
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        a = [torch.nn.Parameter(torch.randn(17, 33)), torch.nn.Parameter(torch.randn(33, 56))]
        torch.nn.init.kaiming_uniform_(a[0], mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(a[1], mode='fan_in', nonlinearity='relu')
        self.modu1 = torch.nn.Linear(in_features=17, out_features=33, bias=False)
        self.acti1 = torch.nn.ReLU()
        self.modu2 = torch.nn.Linear(in_features=33, out_features=56, bias=False)
        self.acti2 = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.modu1(x1)
        v2 = self.acti1(v1)
        v3 = self.modu2(v2)
        v4 = self.acti2(v3)
        return v1, v2, v3, v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 17)
