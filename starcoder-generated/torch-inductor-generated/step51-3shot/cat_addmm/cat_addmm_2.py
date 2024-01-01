
class Model(nn.Module):
    def __init__(self, hidden_dim, input_shape=None):
        super(Model, self).__init__()
        self.input_tensor = torch.randn(input_shape)
        self.linear1 = nn.Linear(2, hidden_dim, bias=False)
        self.linear2 = torch.nn.Linear(3, 2)
    def forward(self, x):
        x = self.input_tensor
        x = self.linear1(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x = torch.randn(3,2)
m = Model(2, (3, 2))
