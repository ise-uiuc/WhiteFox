 from alexnet.py in NVIDIA's torchvision source code
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 256, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            torch.nn.LocalResponseNorm(size=5, alpha=0.001 / 9.0, beta=0.75, k=2),
            torch.nn.Conv2d(256, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            torch.nn.LocalResponseNorm(size=5, alpha=0.001 / 9.0, beta=0.75, k=2),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=6272, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(in_features=4096, out_features=1000),
        )
        
        self.conv = torch.nn.Conv2d(6272, 338, 5, stride=1, padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(338, 1, 2, stride=2, padding=0, output_padding=0)

    def forward(self, x1):
        x2 = self.features(x1)
        x3 = x2.view(x2.size(0), 6272)
        x4 = self.classifier(x3)
        x5 = self.conv(x3)
        x6 = x5 * 0.5
        x7 = x5 * 0.7071067811865476
        x8 = torch.erf(x7)
        x9 = x8 + 1
        x10 = x6 * x9
        x11 = self.conv2(x10)
        return x11
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
