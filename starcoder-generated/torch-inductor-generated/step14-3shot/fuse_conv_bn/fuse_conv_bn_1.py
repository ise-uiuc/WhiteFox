
class Model(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        super().__init__()
        self.conv = torch.nn.Conv1d(15, 1, 3, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm1d(1, track_running_stats=False)
        torch.manual_seed(0)
        self.fc = torch.nn.Linear(10, 20, bias=False)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        features = x.permute(0, 2, 1).contiguous().view(-1, 10)
        return torch.sigmoid(self.fc(features))
# Inputs to the model
input = torch.randn(1, 15, 6)
