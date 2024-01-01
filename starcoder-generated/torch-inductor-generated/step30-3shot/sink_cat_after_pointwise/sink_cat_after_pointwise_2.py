
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Linear(3, 5)
    def forward(self, x):
        t1 = torch.cat((x, x), dim=1)
        t2 = t1.view(x.shape[0], -1).tanh()
        t3 = t2.view(x.shape[0], -1)
        t4 = torch.relu(t3)
        t5 = t4.view(x.shape[0], -1).sigmoid()
        t6 = self.t(t5)
        x = t6
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
