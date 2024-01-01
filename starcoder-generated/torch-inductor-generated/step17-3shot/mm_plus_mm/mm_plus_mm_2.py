
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=(3,3), stride=1),
            torch.nn.Conv2d(16, 16, kernel_size=(1,1), stride=1),
            torch.nn.Conv2d(16, 16, kernel_size=(3,3), stride=1),
            torch.nn.Conv2d(16, 16, kernel_size=(3,3), stride=1),
            torch.nn.Conv2d(16, 16, kernel_size=(3,3), stride=1),
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x
# Inputs to the model
input_shape = (2, 3, 224, 224)
x = torch.randn(input_shape)
output = Model()
y = torch.onnx.export(output, x,"alexnet.onnx", verbose=False, input_names=["input"], output_names=["output"])
