
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 128)
y = m(x1)
__output1__ = y[0][0].item()
__output2__ = y[-1][0].item() 
#__output3__ = y[:, 0].tolist() # TODO: the output should be a list of floating numbers instead of a list of list of floating numbers. To be verified.
#__output4__ = torch.add(x, x)
__output5__ = m(torch.zeros(1,128)).mean(0).item()
__output6__ = torch.dot(x1[-1], y[-1]) / x1.size()[0] # TODO: should be float, not tensor
__output7__ = y[:, int(x1.shape[1]/2)].tolist()
__output8__ = m(x1[0].unsqueeze(0)) # TODO: should be float, not tensor
__output9__ = torch.split(x1, [1, 1]) # TODO: should return a tuple of tensors
__output10__ = m(x1[:, 1].unsqueeze(1).unsqueeze(2)) # TODO: should return a tensor with shape (32, 1, 1, 256), instead of (1, 1, 256)
__output11__ = torch.max(torch.cat((torch.stack([m(x1[0].unsqueeze(0))]*x1.size()[0], dim=0), m(x1)), dim=0), dim=0) # TODO: should return a tuple (tensor_x, tensor_y) with shape (32, 256)
__output12__ = m(x1).squeeze(1) # TODO: should return a tensor with shape (32, 256), instead of (32, 1, 1, 256)

x1 = torch.randn(1, 3, 64, 64)
__output13__ = m(x1) # should contain "conv"