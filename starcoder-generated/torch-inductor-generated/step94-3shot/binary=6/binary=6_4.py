
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_trans = torch.nn.Linear(288, 48)
        self.linear_trans_2 = torch.nn.Linear(107, 27)
 
    def forward(self, x1, x2):
        t1 = self.linear_trans(x1)
        v = t1 - x2
        f1 = self.linear_trans_2(v)
        return f1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 288)
x2 = torch.randn(2, 27)
