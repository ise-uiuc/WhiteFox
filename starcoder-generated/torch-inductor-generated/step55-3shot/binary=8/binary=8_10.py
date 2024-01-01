
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    def forward(self, input_tensor):
        return self.conv1(input_tensor).reshape(input_tensor.size(0),-1).mm(torch.randn((512, 512)))
    def forward(self, input_tensor, arg):
        temp = self.conv1(input_tensor).reshape(input_tensor.size(0),-1)
        # temp2 = torch.randn((512, 512))
        return arg.mm(temp.t())
    def forward(self, input_tensor, arg, kwarg=None):
        temp = self.conv1(input_tensor).reshape(input_tensor.size(0),-1)
        # temp2 = torch.randn((512, 512))
        return (arg.mm(temp.t()))
    def forward(self, input_tensor, arg, kwarg=None):
        temp = self.conv1(input_tensor).reshape(input_tensor.size(0),-1)
        temp2 = torch.randn((512, 512))
        return arg.mm(temp.t()) + arg.mm(temp2.t())
# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
