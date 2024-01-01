
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_out1 = torch.nn.ConvTranspose2d(3, 16, 4, padding=1, stride=2)
        self.conv_in1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv_out2 = torch.nn.ConvTranspose2d(16, 32, 4, padding=1, stride=2)
        self.conv_in2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv_out3 = torch.nn.ConvTranspose2d(32, 128, 4, padding=1, stride=2)
        self.conv_in3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(128, 10)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_out1(x1)
        v2 = self.relu(v1)
        v3 = self.conv_in1(v2)
        v4 = self.relu(v3)
        v5 = self.conv_out2(v4)
        v6 = self.relu(v5)
        v7 = self.conv_in2(v6)
        v8 = self.relu(v7)
        v9 = self.conv_out3(v8)
        v10 = self.relu(v9)
        v11 = self.conv_in3(v10)
        
        v12 = torch.relu(v11)

        # Here flatten op is missing
        v13 = v12

        v14 = self.flatten(v13)
        # Here Linear op is missing
        v15 = torch.softmax(v14, dim=-1)

        return v15

# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
