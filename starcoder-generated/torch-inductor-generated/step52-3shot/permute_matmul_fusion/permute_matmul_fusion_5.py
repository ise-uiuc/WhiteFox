
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.bmm(x1,x2)
    def __call__(self, X, Y):
        return(super(Model, self).__call__(X, Y))
# (Inputs to the model)
X = random_list(1, 2, 32, 16)
Y = random_list(1, 2, 16, 8)
