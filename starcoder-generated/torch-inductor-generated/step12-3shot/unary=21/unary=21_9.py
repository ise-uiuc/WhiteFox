
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return F.tanh(x)
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return nn.Tanh()(x)
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x)
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return x.tanh()
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return F.hardtanh(x)
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return F.hardtanh(x,min_val=-1,max_val=1,inplace=True)
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        y = torch.relu(x, inplace=True)
        return y.tanh()
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        y = x.relu()
        return y.tanh()
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return torch.nn.Tanh()(x)
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return x.tanh_()
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return torch.tanh_(x)
# Model Ends


# Model Beings
class model(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super().__init__()
        self.pool1 = torch.nn.AvgPool2d(4,4,padding=0)
        self.pool2 = torch.nn.AvgPool2d(2,2,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = torch.sigmoid(x)

        return x
