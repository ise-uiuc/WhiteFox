
class Model(torch.nn.Module):        
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(4, 16, 3, stride=1, padding=1, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(16, 32, 3, stride=1, padding=1, output_padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1, output_padding=1)
        self.linear1 = torch.nn.Linear(128, 256, bias=True)
        self.linear2 = torch.nn.Linear(256, 64, bias=True)
        self.linear3 = torch.nn.Lineari(64, 4, bias=True)

    def forward(self, x):
        v10 = self.conv1(x)
        v11 = F.relu(v10, inplace=True)
        v11 = self.conv2(v11)
        v12 = F.relu(v11, inplace=True)
        v12 = self.conv3(v12)
        v12 = F.relu(v12, inplace=True)
        v13 = v12.view((-1, 128))
        v14 = self.liner1(v13)
        v15 = F.sigmoid(v14)
        v15 =  self.liner2(v15)
        v15 = F.sigmoid(v15)
        v15 = self.linear3(v15)
        return v15
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
