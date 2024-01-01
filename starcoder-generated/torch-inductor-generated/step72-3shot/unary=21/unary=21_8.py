
class ModelWithTanh(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)

        self.conv2 = torch.nn.Conv2d(4, 5, 1, stride=1, padding=1)

        self.tanh = torch.nn.Tanh()

    def forward(self, x1):

        # t1 = torch.mean(torch.max(torch.mean(x1, 4), 3))

        # t2 = torch.mean(torch.max(torch.mean(self.tanh(self.conv1(self.conv2(x1))), 4), 3))
        
        t1 = torch.abs(torch.tanh(torch.add(self.conv1(torch.add(self.conv2(x1), 1.0)), 2.0)))

        t2 = torch.abs(torch.tanh(torch.add(self.conv1(torch.tanh(torch.add(self.conv2(x1), 0.1))), 3.0)))

        t3 = torch.abs(torch.tanh(torch.add(self.conv1(torch.tanh(torch.add(self.conv2(x1), 0.1))), 3.0)))
        # t4 = torch.min(torch.max(torch.mean(self.conv1(torch.tanh(torch.add(self.conv2(x1), 0.1))), 4), 3))
        # t5 = torch.min(torch.max(torch.mean(self.conv1(torch.tanh(self.conv2(x1))), 4), 3))
        # t6 = torch.min(torch.mean(self.conv1(torch.max(torch.tanh(self.conv2(x1)), 4)), 3))
        # t7 = torch.sigmoid(torch.tanh(torch.add(self.conv1(torch.tanh(torch.add(self.conv2(x1), 0.1))), 3.0)))
        # return t4

        # t8 = t2*t7

        return t3

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
