
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(4, 2)
        v3 = lstm1(v2)
        linear1 = torch.nn.Linear(2, 2)
        v3 = torch.nn.functional.linear(v3, linear1.weight, linear1.bias).permute(0, 2, 1)
        lstm2 = torch.nn.LSTMCell(4, 2)
        v4 = lstm2(v3)
        linear2 = torch.nn.Linear(2, 2)
        v5 =linear2(v4)
        return v5.permute(0, 2, 1)
# Inputs to the model
x2 = torch.randn(1, 3, 2) 
