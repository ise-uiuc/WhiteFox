
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        x = self.relu(x1)
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
