
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)

    def forward(self, x1, x2):
        t1 = self.linear(x1)
        t2 = t1 + x2
        return t2

# Initialize the model
m = Model()

# Initialize the input tensor
x1 = torch.randn(4, 128)
x2 = torch.randn(4, 128)
