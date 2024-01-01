
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(in_features=1024, out_features=10)
	def forward(self, x1):
        v1 = self.conv(x1)
        v1_resized = (v1[:, :, 1:, 1:] + v1[:, :, :-1, 1:] + v1[:, :, 1:, :-1] + v1[:, :, :-1, :-1]).unsqueeze(1)
        v2 = v1_resized.view([v1_resized.shape[0], -1])
        v3 = self.fc(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
