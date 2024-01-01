
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 2)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.sigmoid(v1)
        return v2
m = Model()
x2 = torch.randn(1, 128)
