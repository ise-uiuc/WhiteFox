
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(64, 64, (1,3), stride=(2,3), padding=(2,1), groups=64, bias=False)
    def forward(self, x5):
        x5_relu = x5.relu()
        x5_relu[:,:,2,2] = x5_relu[:,:,2,2]
        output = self.conv(x5_relu)
        return output
# Inputs to the model
x5 = torch.tensor([[[[-0.7907, -0.5270], [-0.4080,  0.0615]],
                    [[ 0.6802, -0.9245], [-0.6617, -0.0309]],
                    [[ 0.7812,  0.0278], [-0.4741, -0.0843]]],
                   [[[ 0.3115, -1.0310], [ 0.4992, -1.5858]],
                    [[-1.1727, -1.0500], [ 0.2780,  0.5089]],
                    [[-0.0731, -0.7497], [-0.9308, -0.7203]]]])
