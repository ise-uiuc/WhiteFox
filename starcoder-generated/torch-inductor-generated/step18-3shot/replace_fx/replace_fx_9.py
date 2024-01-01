
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    def forward(self, z):
        t1 = torch.nn.functional.dropout(z, p=0.3)
        t2 = torch.rand_like(z)
        t3 = torch.nn.functional.dropout(z, p=0.4)
        t4 = torch.nn.functional.dropout(z, p=0.3)
        t5 = torch.nn.functional.dropout(z, p=0.2)
        return t1 + t2 + t3 + t4 + t5
# Inputs to the model
z1 = torch.zeros([1, 2])
