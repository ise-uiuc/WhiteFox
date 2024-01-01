
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(100 * 100, 2)
       
    def forward(self, x):
        a1 = self.linear(x)
        a2 = torch.sigmoid(a1) 
        return a2

