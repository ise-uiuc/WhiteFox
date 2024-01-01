
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=-1)
        self.linear_1 = torch.nn.Linear(8*8*5, 64)
        self.linear_2 = torch.nn.Linear(64, 64)
        self.linear_3 = torch.nn.Linear(64, 10, dtype=torch.float)
    def forward(self, x):
        x1 = x.to(torch.float)
        x2 = self.flatten(x1)
        x3 = self.linear_1(x2)
        x4 = self.tanh(x3)
        x5 = self.linear_2(x4)
        x6 = self.linear_3(x5)
        x6 = torch.tanh(x6)
        return x6
# Inputs to the model
x = torch.randn(70, 8, 8, 5)
