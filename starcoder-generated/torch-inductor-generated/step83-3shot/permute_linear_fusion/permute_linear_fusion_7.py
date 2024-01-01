
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = torch.sigmoid(self.linear(self.relu(x1)))
        v2 = torch.nn.functional.max_pool2d(v1, 3, padding=0, stride=1, dilation=1, ceil_mode=False)
        v3 = self.sigmoid(v1) + self.relu(v1)
        v4 = torch.nn.functional.elu(v1, alpha=1, inplace=True)
        v5 = self.sigmoid(v1)
        v6 = v4.numel()
        v7 = v4.new_full(size=(1, 2, 3, 3), fill_value=v2)
        return self.relu(v1 + v7) + self.linear(v3 + v5) + torch.sigmoid(self.relu(v6)) + v4
# Inputs to the model
x1 = torch.Tensor(1, 2, 5, 5)
