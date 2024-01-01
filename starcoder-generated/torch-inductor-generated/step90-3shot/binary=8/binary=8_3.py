
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Dropout2d(p=0.5), torch.nn.BatchNorm2d(64), torch.nn.Conv2d(64, 32, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 16, 1, stride=1, padding=0), torch.nn.ReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(16), torch.nn.Conv2d(16, 32, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 2, 1, stride=1, padding=0), torch.nn.ReLU(),
        )

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        return self.layer3(f2)

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer(x)
        out = torch.nn.functional.max_pool2d(out, 2)
        out = torch.nn.functional.dropout2d(out, 0.5, True, False)
        out = torch.nn.functional.batch_norm(out)
        return out

class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64, 0.01, False, False), torch.nn.Conv2d(64, 32, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 2, 1, stride=1, padding=0), torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1), torch.nn.LayerNorm([32, 28, 28], 1.0, 0.0, False), torch.nn.BatchNorm2d(32, 0.01, False), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.LayerNorm([64, 13, 13], 1.0, 0.0, False), torch.nn.BatchNorm2d(64, 0.01, False), torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Model5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 13, 1, 6, 1, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 1, 6, 1, 1, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Model6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 13, 1, 6, 2, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 2, 6, 2, 1, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Model7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 13, 2, 6, 2, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 2, 6, 1, 1, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Model8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 13, 3, 6, 1, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 1, 6, 1, 1, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Model9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 13, 4, 6, 1, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 1, 6, 1, 1, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class ModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 13, 5, 6, 1, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 1, 5, 1, 1, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class ModelB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 14, 4, 6, 2, 1, False), torch.nn.ConvTranspose2d(3, 3, 13, 2, 5, 2, 2, False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


x = torch.randn(1, 2, 64, 64)
