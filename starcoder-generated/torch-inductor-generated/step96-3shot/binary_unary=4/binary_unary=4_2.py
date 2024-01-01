
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        t1 = self.linear1(x)
        t2 = self.linear2(t1 + other)
        t3 = self.linear3(t2)
        return t3

# Initializing the model
m = Model(Tensor(1, 256))

# Inputs to the model
x = torch.randn(1, 512)
