
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 2, 1)
        self.conv2 = torch.nn.Conv1d(2, 2, 1)
        self.conv3 = torch.nn.Conv1d(2, 2, 1)
        self.conv4 = torch.nn.Conv1d(2, 2, 1)
        self.conv5 = torch.nn.Conv1d(2, 2, 1)
        self.conv6 = torch.nn.Conv1d(2, 2, 1)
        self.conv7 = torch.nn.Conv1d(2, 2, 1)
    def forward(self, input):
        x = torch.nn.functional.relu(self.conv1(input))
        out = self.conv6(torch.nn.functional.relu(self.conv2(x)))
        out = self.conv3(torch.nn.functional.relu(self.conv4(out)))
        out = self.conv5(torch.nn.functional.relu(self.conv7(out)))
        x = self.conv2(torch.nn.functional.relu(self.conv1(input)))
        x = self.conv7(torch.nn.functional.relu(self.conv6(x)))
        x = self.conv4(torch.nn.functional.relu(self.conv1(x)))
        out = self.conv3(torch.nn.functional.relu(self.conv5(out)))
        return out
# Inputs to the model
input = torch.randn(3, 1, 20)
torch.manual_seed(0)
