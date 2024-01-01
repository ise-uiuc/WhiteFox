
class Model(torch.nn.Module):
    # [name: l1]
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2, 2, 3)
        # [name: l2]
        self.batch_norm = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v3 = self.relu(self.batch_norm(self.conv2d(x).flatten(start_dim=1).view(1, 2, 4)))
        return v3
# Inputs to the model
x = torch.randn(1, 2, 1, 1)
