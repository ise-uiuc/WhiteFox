
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=26*26, out_features=84)
        self.l2 = torch.nn.Linear(in_features=84, out_features=62)
        self.l3 = torch.nn.Linear(in_features=62, out_features=26*26)
    def forward(self, x):
        x1 = torch.tanh(self.l1(x))
        x2 = torch.tanh(self.l2(x1))
        x3 = torch.tanh(self.l3(x2))
        return x3.reshape((x3.shape[0], 26, 26))
# Inputs to the model
x = torch.randn(1, 26*26)
