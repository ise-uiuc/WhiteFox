
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.linear2 = torch.nn.Linear(2, 1, bias=False)
        self.linear3 = torch.nn.Linear(2, 1, bias=False)
    def forward(self, x0):
        x1 = torch.transpose(x0, 1, 2)
        x2 = torch.reshape(x1, (-1, x1.size()[1]))
        v1 = torch.matmul(x0, self.linear.weight)
        v2 = v1 + self.linear.bias
        v1 = v2 + self.linear2.weight
        v2 = torch.unsqueeze(v1, dim=1)
        v1 = torch.squeeze(v2, dim=1)
        v2 = v1 + self.linear3.weight
        return v2
# Inputs to the model
x0 = torch.randn((1, 2, 2))
