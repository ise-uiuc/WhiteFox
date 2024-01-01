
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(7184, 512)
        self.batch_norm1 = torch.nn.BatchNorm1d(512, eps=1e-05, track_running_stats=True)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.fc1.weight, self.fc1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.batch_norm1.weight, self.batch_norm1.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 7184, 2)
