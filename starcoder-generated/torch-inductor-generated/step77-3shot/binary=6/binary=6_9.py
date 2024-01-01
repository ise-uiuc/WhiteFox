
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128, bias=True)

    def forward(self, t1, t2):
        v4 = self.linear(t1)
        v5 = v4 - t2
        return v5
# Initialization of the model
m = Model()

# Input tensors to the model
t1 = torch.randn(128, 128)
t2 = torch.randn(128)
