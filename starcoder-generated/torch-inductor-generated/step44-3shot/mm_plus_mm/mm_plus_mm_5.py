
class Model(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
    def forward(self, x):
        x = x
        t1 = torch.zeros(x.size()[0], self.num_classes).cuda()
        return t1
# Inputs to the model
x = torch.randn(1, 1, 14, 14)
