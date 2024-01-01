
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        hidden1 = x1.permute(0, 2, 1)
        hidden2 = self.layer0(hidden1)
        hidden3 = x2.permute(0, 2, 1)
        hidden4 = self.layer0(hidden3)
        v1 = torch.bmm(hidden2, hidden4)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
