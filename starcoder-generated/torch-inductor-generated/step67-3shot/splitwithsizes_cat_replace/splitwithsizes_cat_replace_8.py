
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Sequential(OrderedDict(
                [('0', torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)),
                 ('1', torch.nn.BatchNorm2d(32))]
            ))),
            ('conv2', torch.nn.Sequential(OrderedDict(
                [('0', torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)),
                 ('1', torch.nn.BatchNorm2d(32))]
            ))),
            ('conv3', torch.nn.Sequential(OrderedDict(
                [('0', torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)),
                 ('1', torch.nn.BatchNorm2d(32))]
            )))
        ]))
        self.decoder = torch.nn.Sequential(OrderedDict([
            ('relu1', torch.nn.ReLU()),
            ('upconv1', torch.nn.Sequential(OrderedDict(
                [('0', torch.nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=False)),
                 ('1', torch.nn.BatchNorm2d(32)),
                 ('2', torch.nn.ReLU())]
            ))),
            ('upconv2', torch.nn.Sequential(OrderedDict(
                [('0', torch.nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=False)),
                 ('1', torch.nn.BatchNorm2d(32)),
                 ('2', torch.nn.ReLU())]
            ))),
            ('conv4', torch.nn.Conv2d(32, 3, 3, 1, 1, bias=False))
        ]))
        self.final = torch.nn.Sequential(OrderedDict([
            ('norm1', torch.nn.BatchNorm2d(32))
        ]))
    def forward(self, v1):
        split_tensors = torch.split(v1, [3, 3, 3, 3], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        concatenated_tensor2 = self.decoder(self.final(concatenated_tensor))
        return (concatenated_tensor2, torch.split(v1, [3, 3, 3, 3], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
