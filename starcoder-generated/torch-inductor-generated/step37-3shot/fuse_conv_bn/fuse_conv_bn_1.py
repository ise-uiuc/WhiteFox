
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32, momentum=0.1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32, momentum=0.1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32, momentum=0.1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 8),
        )
    def forward(self, x):
        return self.features(x)
# Inputs to the model
x = torch.randn(1, 1, 30)
