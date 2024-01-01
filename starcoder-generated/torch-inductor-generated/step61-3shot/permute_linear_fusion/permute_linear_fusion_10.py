
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 5)
        self.linear2 = torch.nn.Linear(5, 5)
    def forward(self, x1):
        x1 = x1.permute(0, 3, 2, 1)
        x2 = torch.nn.functional.relu(self.linear1(x1))
        x3 = torch.nn.functional.linear(x2, self.linear2.weight, self.linear2.bias)
        x4 = x3.permute(0, 2, 3, 1)
        x5 = torch.nn.functional.relu(x4)
        x6 = x5.permute(0, 3, 1, 2)
        x7 = torch.nn.functional.linear(x6, torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]).permute(3, 1, 0), torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]).permute(3, 1, 0))
        x8 = torch.nn.functional.sigmoid(x7)
        return x8
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
