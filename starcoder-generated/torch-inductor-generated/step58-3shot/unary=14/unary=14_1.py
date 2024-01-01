
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 7, stride=2, padding=3, bias=True)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(1, 1, ((1, 1), (3, 2)), stride=2, padding=1, bias=True)
    def forward(self, input1):
        v1 = self.conv(input1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_5(v3)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.max(v5)
        v7 = torch.topk(v5, k=1)
        v8 = torch.unsqueeze(v6, dim=0)
        v9 = v8.expand(1, 2, 3, 4)
        v10 = torch.nn.functional.normalize(v8, p=1, dim=None)
        v11 = v10 * v9
        return v6
# Inputs to the model
input1 = torch.randn(1, 2, 8, 8)
