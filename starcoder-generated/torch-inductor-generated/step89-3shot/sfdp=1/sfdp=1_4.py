
class Model(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_in = torch.nn.Linear(2, self.d_model)

    def forward(self, x1, x2):
        v1 = self.linear_in(x2)  # q: query; k: key; v: value
        output = torch.matmul(x1, v1.transpose(-2, -1))
        return output

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(1, 2, 64)
x2 = torch.randn(1, 64, 2)
