
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = x
        t2 = x
        for i in range(5):
            t1 = torch.cat((t1, t1, t1, t1), dim=1)
            t1 = t1.view(t1.shape[0], -1)
            t1 = torch.relu(t1)
            t2 = torch.cat((t2, t2, t2, t2), dim=1)
            t2 = t2.view(t2.shape[0], -1)
            t2 = torch.relu(t2)
        x = t1 + t2
        return x.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
