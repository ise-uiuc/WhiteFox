
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    def forward(self, z):
        t1 = torch.nn.functional.dropout(z, p=0.3)
        t2 = torch.rand_like(z)
        return t2 + t1
# Inputs to the model
z1 = torch.zeros([1, 2])
