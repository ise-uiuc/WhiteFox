
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,1, stride=1)
        self.conv2 = nn.Conv2d(32,16,1, stride=1)
        self.conv3 = nn.Conv2d(16,8,1, stride=1)

        self.tanh1 = nn.Hardtanh(min_val=0, max_val=255)
        self.tanh2 = nn.Hardtanh(min_val=0, max_val=255)
        self.tanh3 = nn.Hardtanh(min_val=0, max_val=255)

    def forward(self, x):
        t1 = self.conv1(x)
        t1 = t1.transpose(2,3)
        
        t2 = self.conv1(t1)
        t2 = self.conv2(t2)
        t2 = t2.transpose(2,3)
        
        t3 = self.conv2(t2)
        t3 = self.conv3(t3)
        t3 = t3.transpose(2,3)

        t4 = self.conv3(t3)
        return self.tanh3(t4)
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
