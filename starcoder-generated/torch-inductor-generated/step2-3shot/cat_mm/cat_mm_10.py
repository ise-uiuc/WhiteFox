
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.cnd = nn.Linear(1,1,bias=True)
        self.cnd.weight.data.fill_(0.25)
        self.cnd.bias.data.fill_(0.0)
    def forward(self, x1, x2):
        a1=self.fc1(x1)
        b1=self.fc1(x2)
        a2=torch.mm(a1,b1)
        a3=self.cnd(a2)
        a4=a3+a3
        a4=a4+a3
        return a4
x1 = torch.FloatTensor([[1,1],[1,1]])
x2 = torch.FloatTensor([[1,1],[0,0]])

from torchsummary import summary
summary(Model(), (2,2))