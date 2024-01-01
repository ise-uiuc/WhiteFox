
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v2 * 2
        v3 = v2.unsqueeze(1)
        v4 = self.batch_norm(v3)
        v5 = v4.squeeze(1)
        v6 = v2 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
