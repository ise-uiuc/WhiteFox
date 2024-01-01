
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        conv_out = self.conv1_1(x)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv1_2(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv1_3(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv2_1(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv2_2(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv2_3(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv3_1(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv3_2(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv3_3(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv4_1(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv4_2(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv4_3(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv5_1(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv5_2(conv_out)
        conv_out = nn.ReLU(conv_out)
        conv_out = self.conv5_3(conv_out)
        conv_out = nn.ReLU(conv_out)
        return conv_out
# Inputs
x = torch.randn(1, 3, 224, 224)
# Generated model ends

# Sample test begins
def test(a):
    model = Model()

    x1 = torch.randn(7, 1, 28, 28) # mnist shape
    x2 = torch.randn(7, 3, 224, 224) # imagenet shape
    x3 = torch.randn(7, 3, 32, 32) # kaggle MNIST shape from https://www.kaggle.com/hocop1/kaggle-intro

    # Forward
    v1_1 = model(x1) # mnist
    v2_1 = model(x2) # imagenet
    v3_1 = model(x3) # kaggle MNIST

    # Check that the output channels = num_filters
    assert v1_1.shape == (7, 16, 14, 14)
    assert v2_1.shape == (7, 512, 7, 7)
    assert v3_1.shape == (7, 512, 28, 28)

    # Check if the number of parameters > 10000 and < 20000
    assert len(list(model.parameters())) >= 10000
    assert len(list(model.parameters())) < 20000

    print("Congratulations! Your generated model has passed all our tests!")
# Sample test ends
