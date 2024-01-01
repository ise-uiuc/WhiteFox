
class MyModule(torch.nn.Module):
    def __init__(self, weight):
        super(MyModule, self).__init__()
        self.conv = torch.nn.Conv2d(1, 20, 5, 1)
        self.linear_1 = torch.nn.Linear(20, 10)
        self.linear_2 = torch.nn.Linear(10, 1)
        self.weight = torch.nn.Parameter(weight)

    def forard(self, x):
        out = self.conv(x)
        out = self.linear_1(out.view(out.size(0), -1))
        out = out.relu()
        out = self.linear_2(out.view(out.size(0), -1))
        out = torch.sigmoid(out)
        out = torch.add(out, self.weight)
        return out

# Inputs to the model
x = torch.rand(1,1,28,28)
