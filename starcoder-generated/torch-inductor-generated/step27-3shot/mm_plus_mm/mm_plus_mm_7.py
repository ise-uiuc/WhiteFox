
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.d1 = torch.nn.Linear(in_features=256, out_features=16)
        self.d2 = torch.nn.Linear(in_features=16, out_features=6)
        self.d3 = torch.nn.Linear(in_features=16, out_features=16)
        self.d4 = torch.nn.Linear(in_features=6, out_features=16)
        self.d5 = torch.nn.Linear(in_features=16, out_features=16)
        self.d6 = torch.nn.Linear(in_features=16, out_features=16)
        self.d7 = torch.nn.Linear(in_features=16, out_features=16)
        self.d8 = torch.nn.Linear(in_features=16, out_features=1)
    def forward(self,x1,x2,x3,x4):
        hidden1 = self.d2(self.d1(x2))
        hidden2 = self.d3(self.d4(hidden1))
        hidden3 = self.d5(self.d6(self.d7(hidden2)))
        hidden3 = self.d5(self.d6(self.d7(hidden2)))
        hidden4 = self.d1(self.d8(x3))
        hidden5 = self.d1(self.d8(x1))
        hidden6 = torch.mm(hidden4, hidden5)
        return hidden6
# Inputs to the model
x1 = torch.randn(18,256)
x2 = torch.randn(18,256)
x3 = torch.randn(25,1)
x4 = torch.randn(25,6)
