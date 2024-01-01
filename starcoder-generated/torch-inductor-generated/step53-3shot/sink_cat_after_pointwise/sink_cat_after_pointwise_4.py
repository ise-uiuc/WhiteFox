
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1
    def forward(self, x):
        y = x.tanh()
        if y.dim() == 2:
            y = y.reshape(y.shape[0],-1)
        # print(y.shape)
        x = torch.cat(2*[y], dim=1)
        # print(x.shape)
        if x.dim() == 3:
            x = x.view(x.shape[0],-1)
        # print(x.shape)
        # self.a = 1000
        if y.shape!= (12, 4):
            x = self.fc_tanh(x)
        else:
            print(self.a)
            x = self.fc_relu(x)
        return x
    def fc_tanh(self, x):
        return torch.tanh(x)
    def fc_relu(self, x):
        return torch.relu(x)
# Inputs to the model
x = torch.randn(2, 3, 4)
