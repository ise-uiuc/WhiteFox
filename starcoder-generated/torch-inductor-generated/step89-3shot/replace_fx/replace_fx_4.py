
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.dropout1 = torch.nn.Dropout(0.2)
    def forward(self, x):
        x = self.dropout1(x)
        x = torch.rand_like(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
