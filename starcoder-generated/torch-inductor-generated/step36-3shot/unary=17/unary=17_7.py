
module_2 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1)
module_1 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1)
model = torch.nn.Sequential(
    module_1,
    torch.nn.ReLU(),
    module_2,
    torch.nn.ReLU(),
)
# Input to the model
x1 = torch.randn(1, 4, 64, 64)
# model definition ends
