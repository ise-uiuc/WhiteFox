
class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 32 @ 224 x 224
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 32 @ 224 x 224
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), # 64 @ 224 x 224
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True), # 64 @ 112 x 112
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 128 @ 112 x 112
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 128 @ 112 x 112
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True), # 128 @ 56 x 56
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 256 @ 56 x 56
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 256 @ 56 x 56
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 256 @ 56 x 56
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 256 @ 56 x 56
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True), # 256 @ 28 x 28
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True), # 256 @ 14 x 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True), # 512 @ 14 x 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True), # 512 @ 14 x 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True), # 512 @ 14 x 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True), # 512 @ 14 x 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True), # 512 @ 14 x 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1), nn.ReLU(inplace=True), # 512 @ 14 x 14
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 1024 @ 14 x 14
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 1024 @ 14 x 14
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
    def forward(self, x1, x2):
        x = self.features(x1)
        if x2 is not None:
            x = torch.cat([x, x2], dim=1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 1024, 1, 1)
