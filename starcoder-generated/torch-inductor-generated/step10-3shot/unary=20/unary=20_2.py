
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trconv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=128, kernel_size=(3, 3))
        self.dropout = torch.nn.Dropout(0.2)
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten(1, 3)
        self.linear = torch.nn.Linear(in_features=1, out_features=1, bias=False)
    def forward(self, x):
        x1 = self.trconv(x)
        x = self.sigmoid(x1)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = self.linear(x)
        return x
# Input to the model
x1 = torch.randn(1, 3, 64, 64)
