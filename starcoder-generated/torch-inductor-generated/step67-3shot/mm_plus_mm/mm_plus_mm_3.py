
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, i1, i2, i3, i4, i5):
        t1  = torch.matmul(i1, i4)
        t2  = torch.matmul(i2, i3)
        s1  = torch.sigmoid(t1)
        s2  = torch.sigmoid(t2)
        s3  = torch.sigmoid(t1)
        s4  = torch.sigmoid(t2)
        t3 = torch.cat([s3, s4], dim=0)
        t4 = torch.cat([s1, s2], dim=0)
        s5 = torch.sigmoid(t3)
        s6 = torch.sigmoid(t4)
        return s5 + s6

# Input to the model ends.
