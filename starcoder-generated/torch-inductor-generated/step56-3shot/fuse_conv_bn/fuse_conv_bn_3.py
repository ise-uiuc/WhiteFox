
model = torch.nn.Sequential(torch.nn.Conv2d(64, 8, [3,11], stride=(2, 1), padding=(3, 5), dilation=(2,1)),
                           torch.nn.BatchNorm2d(8),
                           torch.nn.Conv2d(8, 9, [5,11], stride=(3, 1), padding=(4, 5), dilation=(3,1)),
                           torch.nn.BatchNorm2d(9),
                           torch.nn.Flatten(),
                           torch.nn.Linear(18169, 10))                       
# Inputs to the model
x = torch.randn(2, 64, 65, 65)
