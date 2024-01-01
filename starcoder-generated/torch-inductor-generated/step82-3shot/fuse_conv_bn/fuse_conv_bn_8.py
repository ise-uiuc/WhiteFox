
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv2d(4, 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        batchnorm_layer = torch.nn.BatchNorm2d(2, eps=2e-5, momentum=0.1)
        torch.manual_seed(8)
        self.layer = torch.nn.Sequential(conv, batchnorm_layer)
    def forward(self, x2):
        s2 = self.layer(x2)
        return s2
# Inputs to the model
x2 = torch.randn(1, 4, 5, 5)
