
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        def custom_softmax(self, input):
            max_input = torch.max(input, dim=1).values
            max_input = max_input.view(max_input.size(0), 1).expand_as(input)
            input = input -  max_input
            exp = torch.exp(input)
            exp_sum = torch.sum(exp, dim=1).view(exp.size(0), 1).expand_as(exp)
            out = exp / exp_sum
            return out
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.layers(x).tanh()
        x = self.softmax(x)
        return x
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.tensor([[1000., 0.0000001, 0.01, 10., 1000.]] * 3)
