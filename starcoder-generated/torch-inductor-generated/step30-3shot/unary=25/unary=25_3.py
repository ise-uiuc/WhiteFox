
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(89, 1028)
        self.linear2 = torch.nn.Linear(1028, 1152)
        self.fc = torch.nn.Linear(1152, 5)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        t1 = self.linear2(v4)
        t2 = t1 > 0
        t3 = t1 * 0.01
        t4 = torch.where(t2, t1, t3)
        f1 = self.fc(t4)
        return f1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 89)
