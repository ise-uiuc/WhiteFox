
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2, bias=False)
        self.linear3 = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        x2 = x1.permute(0, 2, 1)
        x3 = torch.nn.functional.linear(x2, self.linear1.weight, output_process_bias=None)
        x4 = torch.nn.functional.linear(x2, self.linear2.weight, None)
        x5 = torch.maximum(x4, x3)
        x6 = x5.permute(0, 2, 1)
        x7 = torch.nn.functional.linear(x6, self.linear3.weight, None)
        return x7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
