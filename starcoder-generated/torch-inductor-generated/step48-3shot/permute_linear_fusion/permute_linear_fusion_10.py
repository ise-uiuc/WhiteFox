
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=(0,), dilation=(1,), ceil_mode=False)
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        v4 = v2 + v3
        v5 = self.maxpool(v4)
        v6 = self.softmax(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
