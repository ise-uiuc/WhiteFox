
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.zeros([6, 4])
        t2 = torch.zeros([4, 6])
        t1 = t1 + t2
        t3 = torch.zeros([6, 4])
        v1 = torch.view_as_strided(t1, shape=[3, 2, 2], stride=[1, 2, 2], writeable=True)
        v2 = torch.zeros([1, 1]) + v1[:, 0, 0]
        return v2
# Input to the model
x = torch.zeros([2, 2])
