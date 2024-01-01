
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.conv = Conv2d(3, 3, 3, stride=1, padding=1, groups=3)
     
        # In this toy network example, this constant is the same as the input. Please feel free to change it as you need.
        self.other = torch.tensor([[[[[-1, -1, -1, -1, -1],
                                          [-1, -1, -1, -1, -1],
                                          [-1, -1, -1, -1, -1],
                                          [-1, -1, -1, -1, -1],
                                          [-1, -1, -1, -1, -1]]]]])
 
    def forward(self, x):
        t1 = self.linear(x)
        t2 = t1 - self.other
        t3 = t2.relu()
        t4 = t2.sigmoid()
        t5 = t2.tanh()
        t6 = t1.softmax(dim=1)
        t7 = self.conv(t3)
        return t1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn((1, 2), device='cuda')
