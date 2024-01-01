
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = list(numpy.ndenumerate(v1.data.numpy()))
        v3 = list(map(lambda x:x[1], v2))
        v3 = list(filter(lambda x:x>0, v3))
        v4 = [(i[0], x1[i[0][0]][1][i[0][1]] * 0.2) if i[1]>0 else i for i in list(v2)]
        x2 = numpy.array([map(lambda x:x[1], [i for i in v4 if i[0][0]==i[1][0]]) for i in range(x1.shape[0]) if map(lambda x:x[1], [i for i in v4 if i[0][0]==i[1][0]])])
        x3 = torch.tensor(x2)
        v5 = torch.where(x3 >= 0, torch.Tensor(x2), v1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 1, 20, 1)
