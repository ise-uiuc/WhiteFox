
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 79, 12, stride=2, padding=11)
        self.relu = torch.relu
    def forward(self, x1050):
        v1 = self.conv(x1050)
        v2 = self.relu(v1)
        v3 = v2 * 0.1310812254497309
        v4 = v2 * 0.35263921331165196
        v5 = v2 * 0.17631960665582598
        v6 = v4 + v5
        v7 = v2 * 0.4472135955358208
        v8 = v2 * v6
        v9 = v8 * 0.9992006711707702
        v10 = v3 - v9
        return v10
# Inputs to the model
x1050 = torch.randn(1, 22, 13, 9)
