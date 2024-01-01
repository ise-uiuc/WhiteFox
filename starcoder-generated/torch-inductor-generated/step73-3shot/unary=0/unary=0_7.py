
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 4, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 4, stride=2, padding=1)
	self.linear1 = torch.nn.Linear(32 * 6 * 6, 10)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = self.linear1(v5.view(v5.size(0), -1))
        v7 = torch.log_softmax(v6, dim=1)
        return v7
# Inputs to the model
x1 = torch.tensor(torch.randn([1, 1, 64, 64], dtype=torch.float32))
