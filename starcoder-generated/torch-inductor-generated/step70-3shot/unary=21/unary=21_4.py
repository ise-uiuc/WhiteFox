
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(12, 16, 1, bias=False),
            torch.nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 6, 1, bias=False),
            torch.nn.Sigmoid())
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(1, 12, 400)
