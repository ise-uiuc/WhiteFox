
from torchvision import models
class Model(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.cnn = models.inception_v3(pretrained=pretrained)
        self.fc = torch.nn.Linear(2048, 10)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
    def forward(self, input1, input2=None, input3=None, input4=None):
        t1 = self.cnn.Conv2d_1a_3x3(input1)
        t2 = self.cnn.Conv2d_2a_3x3(input2)
        t3 = self.cnn.Conv2d_2b_3x3(input4)
        t4 = self.cnn.maxpool1(t1+t2)
        t5 = self.cnn.Conv2d_3b_1x1(t4)
        t6 = self.cnn.Conv2d_4a_3x3(t5)
        t7 = self.cnn.maxpool2(t6)
        t8 = self.cnn.Mixed_5b(t7)
        t9 = self.cnn.Mixed_5c(t8)
        t10 = self.cnn.Mixed_5d(t9)
        t11 = self.cnn.Mixed_6a(t10)
        t12 = self.cnn.Mixed_6b(t11)
        t13 = self.cnn.Mixed_6c(t12)
        t14 = self.cnn.Mixed_6d(t13)
        t15 = self.cnn.Mixed_6e(t14)
        t16 = self.cnn.Mixed_7a(t15)
        t17 = self.cnn.Mixed_7b(t16)
        t18 = self.cnn.Mixed_7c(t17)
        t19 = self.avgpool(t18)
        t20 = self.cnn.dropout(t19)
        t21 = self.fc(t20.view(input1.size(0), -1))
        return t21
# Inputs to the model
input1 = torch.randn(1, 3, 299, 299)
input2 = torch.randn(1, 3, 299, 299)
input3 = torch.randn(1, 3, 299, 299)
input4 = torch.randn(1, 3, 299, 299)
