
model = torch.nn.Sequential(torch.nn.Conv2d(5, 10, 3), torch.nn.BatchNorm2d(10), torch.nn.Conv2d(10, 20, 1), torch.nn.ReLU(), torch.nn.BatchNorm1d(20), torch.nn.Dropout2d(0.1500), torch.nn.ConvTranspose2d(20, 20, 2, stride=2, output_padding=1), torch.nn.ReLU6(), torch.nn.BatchNorm1d(20), torch.nn.ConvTranspose2d(20, 5, 1, bias=False), torch.nn.Sigmoid())
# Inputs to the model
x1 = torch.randn(1, 5, 5, 5)
