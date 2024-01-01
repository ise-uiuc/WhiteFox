
class Model(torch.nn.Module):
    def __init__(self, hidden_dim=(1024, 512)):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(7),
            torch.nn.Conv1d(512, 512, 3),
            torch.nn.Conv1d(512, 512, 3),
            torch.nn.Conv1d(512, 512, 3),
            torch.nn.AdaptiveAvgPool1d(7),
            torch.nn.Flatten(),
            torch.nn.Linear(8192, hidden_dim[0]),
            torch.nn.Linear(hidden_dim[0], hidden_dim[0]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim[0], hidden_dim[0]),
            torch.nn.Linear(hidden_dim[0], hidden_dim[0]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim[0], hidden_dim[0]),
        )
    def forward(self, x):
        y = self.module(x)
        y = torch.softmax(y, dim=1)
        return y
# Inputs to the model
x = torch.randn(8, 512, 1024)
