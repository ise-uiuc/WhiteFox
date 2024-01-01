
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1,1)
        self.linear2 = torch.nn.Linear(1,1)
        self.linear3 = torch.nn.Linear(1,1)
        self.linear4 = torch.nn.Linear(1,1)

    def forward(self, input1, input2, input3):
        x1, x2, x3, x4 = input1, input2, input3, input1
        x1 = torch.mm(x1,x4)
        x2 = self.linear1(input1)
        x3 = torch.mm(x2,x1)
        x4 = self.linear2(x3)
        x1 = torch.mm(x2, input2)
        x2 = self.linear3(x1)
        x3 = self.linear3(x2)
        x4 = torch.mm(x3, input3)
        x1 = self.linear3(x3)
        x2 = self.linear4(x1)
        x3 = x4 + x2
        return x2 + x3 + x3
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
