
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        t0, t1, t2, t3 = torch.split(x, [4,2,2,1], 1)
        t4 = torch.cat((t0, t1), 1)
        t5 = torch.cat((t2, t3), 1)
        t6 = torch.cat((t4, t5), 1)
        return t6

# Initializing the model
m = Model()


# Input to the model
x = torch.randn(1, 6, 4, 4)

