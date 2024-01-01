
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(31, 31, 3, stride=2)
        self.dropout = torch.nn.Dropout(p=0.15)
        self.conv2d = torch.nn.Conv2d(122, 63, 1)
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.dropout_4 = torch.nn.Dropout(p=0.034)
        self.dense = torch.nn.Linear(365,10)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv2d(v9)
        v11 = v10.view(-1, 122)
        v12 = self.dropout(v11)
        v13 = self.adaptive_avg_pool2d(v12)
        v14 = self.flatten(v13)
        v15 = self.dropout_4(v14)
        v16 = self.dense(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 31, 200, 300)
