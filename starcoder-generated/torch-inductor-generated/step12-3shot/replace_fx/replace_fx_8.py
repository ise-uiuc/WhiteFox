
class Model(torch.nn.Module):
    def __init__(self, d=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 2)
        self.d = d
    def forward(self, x1):
        x2 = self.fc1(x1)
        x3 = torch.nn.functional.dropout(x2, p=self.d, training=True)
        return x3
# Inputs to the model
x1 = torch.randn(10)
