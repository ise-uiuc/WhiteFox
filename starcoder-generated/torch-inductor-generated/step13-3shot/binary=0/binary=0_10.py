
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, t1, t2, other=-1):
        if t1 == None:
            t1 = torch.randn(t2.shape)
        t1 = self.flatten(t1)
        v2 = t2 - other
        return v2
# Inputs to the model
t2 = torch.randn(2, 3, 2, 5)
