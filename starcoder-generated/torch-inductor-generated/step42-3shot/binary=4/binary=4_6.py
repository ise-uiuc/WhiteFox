
class Model(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(9, 8)

    def forward(self, x1, x2):
        return self.linear1(x1) + x2


# Initializing the model
x1 = torch.randn(3, 9)
x2 = torch.randn(3, 8)
m = Model()
