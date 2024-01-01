
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.in_channels = 3
        self.conv = nn.Conv2d(3, 64, kernel_size=9, stride=1)
        self.bbr = torch.nn.Sequential(nn.ModuleList([MyBBR() for _ in range(0,16)]))
        self.bn = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0) # kernel size, stride
        self.stage1_unit1 = BasicBlock_Custom(64, 64, 1, 1, 1) # in chennel, out channels, kernel size, stride, padding
        self.stage2_unit2 = self._make_transition_layer([256, 128], [256, 128], [True, False], [2, 2]) # transition layers
        self.stage3_unit3 = self._make_stage([3, [256, 128, 256, 128], [256, 256, 256, 256], [False, True, True, False]], 4) # blocks args
        self.stage4_unit4 = self._make_stage([3, [256, 64, 256, 64], [256, 256, 256, 256], [False, True, True, False]], 1)

    def forward(self, x):
        x = self.conv(x)        #[1,256,1,1]
        x = self.bbr(x)         #[1,256,1,1]
        x = self.relu1(self.bn(x))       #[1,256,1,1]
        x = self.maxpool(x)     #[1,256,1,1]
        x = self.stage1_unit1(x) #[1,256,1,1]
        x = self.stage2_unit2(x) #[1,256,2,2]
        x = self.stage3_unit3(x) #[1,256,4,4]
        x = self.stage4_unit4(x) #[1,256,4,4]
        return x
# Inputs to the model
x = Variable(torch.randn(1, 1, 4, 4))
