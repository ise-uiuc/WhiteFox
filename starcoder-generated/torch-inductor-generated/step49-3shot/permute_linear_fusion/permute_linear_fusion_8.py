
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.conv1_1 = torch.nn.Conv1d(in_channels=2, out_channels=8, kernel_size=(22, 4), stride=(6, 2), padding=(11,), dilation=(1, 1))
        self.conv2_1 = torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=(22, 4), stride=(6, 2), padding=(11,), dilation=(1, 1))
        self.relu = torch.nn.ReLU()
        self.max1 = torch.nn.MaxPool1d(4)
        self.max2 = torch.nn.MaxPool1d(2)
        self.flatten = torch.flatten
    def forward(self, x1):
        v1 = self.conv1_1(x1)
        v2 = self.relu(v1)
        v3 = self.max1(v2)
        v4 = self.conv2_1(v3)
        v5 = self.relu(v4)
        v6 = self.flatten(v5, 1)
        v7 = torch.dot(v6, self.linear.weight.T) + self.linear.bias
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 10)
