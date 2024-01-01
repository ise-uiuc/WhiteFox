
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = torch.where(self.relu(x1) > 0, self.tanh(self.linear(self.relu(self.linear(x1)))), torch.zeros_like(self.linear(self.relu(self.linear(x1)))))
        return v1
# Inputs to the model
x1 = torch.randn(10, 1, 4, 4)
