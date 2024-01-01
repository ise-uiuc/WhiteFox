
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d((8, 8))
        self.dense1 = torch.nn.Linear(32*10*10, 512)
        self.drop = torch.nn.Dropout(p=.3)
        self.dense2 = torch.nn.Linear(512, 10)
    def forward(self, x):
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = F.softmax(x, dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
