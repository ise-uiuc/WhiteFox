
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(2, 1)
    @staticmethod
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, torch.nn.functional.relu(self.linear.weight), torch.nn.functional.relu(self.linear.bias))


        z = torch.matmul(v1, torch.nn.functional.relu(self.linear.weight), torch.nn.functional.relu(self.linear.bias))


        y = torch.nn.functional.relu(z)
        return torch.nn.functional.linear(y, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
