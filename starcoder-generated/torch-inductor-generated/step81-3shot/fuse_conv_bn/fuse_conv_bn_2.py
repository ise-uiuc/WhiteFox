

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(131)
        self.layer1 = torch.nn.Conv2d(9, 3, 4)
        torch.manual_seed(13)
        self.layer2 = torch.nn.BatchNorm2d(3)
        torch.manual_seed(113)
        self.layer3 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.63, inplace=True)

    def forward(self, x1):
        # self.layer1.bias is None
        x1 = self.layer3(self.layer2(self.layer1(x1)))
        x1 = self.dropout(x1)
x1 = torch.randn(1, 9, 100, 100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss = model(x1)
for param in model.parameters():
    print(param)
optimizer.step()
# Inputs to the model
