
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,6,3,padding=3,bias=False,stride=3)
        self.conv12 = torch.nn.Conv2d(6,24,1,padding=0,stride=1,bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(1,4,5,stride=5,padding=5,bias=False)
        self.conv22 = torch.nn.ConvTranspose2d(4,4,2,stride=2,padding=2,bias=False)
        self.conv3 = torch.nn.Conv2d(4,8,3,padding=2,stride=2,bias=False)
        self.relu = torch.nn.ReLU()
        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x):
        v20 = 0.7978845608028654
        v19 = 0.044715
        v18 = 0.5
        v15 = self.conv12(x)
        v11 = self.conv2(v15)
        v6 = self.conv3(v11)
        v13 = self.conv22(v15)
        v22 = self.prelu(v6)
        v21 = self.conv1(x)
        v5 = self.conv22(v21)
        v16 = self.relu(v21)
        v9 = v22 * v13
        v12 = v16 + v19
        v7 = v22 * v12
        v8 = v16 + v20
        v10 = v18 * v22
        v14 = v10 * v9
        return v14
# Inputs to the model
x = torch.randn(1, 1, 18, 18)
