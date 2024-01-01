
class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x2 = x1 + 1.0
        x3 = torch.nn.functional.dropout(x2, p=0.3)
        return x3
# Input to the model
inputs, labels = (torch.randn(1, 10, requires_grad=True), torch.randn(1, 10, requires_grad=True))
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand(1)
        x3 = torch.randint(0, 9, (1,))
        x4 = torch.rand_like(x3)
        x5 = torch.nn.functional.dropout(x1)
        x6 = torch.nn.functional.dropout(x2)
        x7 = torch.nn.functional.dropout(x3)
        return x7
# Input to the model
x1 = torch.randn(1)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand(1)
        x3 = torch.randint(0, 9, (1,))
        x4 = torch.rand_like(x3)
        x5 = torch.nn.functional.dropout(x1)
        return x5
# Input to the model
x1 = torch.randn(1)
# Model end
