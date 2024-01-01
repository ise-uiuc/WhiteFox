
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(62, 29, 23, stride=2, padding=2)
        self.dropout = torch.nn.Dropout(p=0.27975041001800917)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.dropout(v1)
        v3 = v2 * 0.7678297169611875
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.2982368326428082
        v7 = v3 + v6
        v8 = v7 * 0.7860601645176003
        v9 = self.relu(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 62, 11, 11)
