
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        conv = nn.Conv1d(16, 33, 3, stride=2, padding=4)
        torch.manual_seed(1)
        bn = nn.BatchNorm1d(num_features=33)
        torch.manual_seed(1)
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv', conv),
            ('bn', bn)
        ]))
    def forward(self, x):
        x = self.model(x)
        return x.squeeze()
# Inputs to the model
x = torch.randn(32, 16, 50)
