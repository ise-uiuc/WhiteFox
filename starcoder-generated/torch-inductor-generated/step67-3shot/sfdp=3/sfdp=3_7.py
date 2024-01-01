
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.d_model = 512
        self.h = 8
        self.w = self.h
        self.conv1 = nn.Conv2d(3, 512, 1)
        self.conv2 = nn.Conv2d(512, self.d_model, 1)
        assert (self.d_model % self.h) == 0
        assert self.d_model == self.w * self.h
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.h]))
 
    def forward(self, x1):
        x2 = self.conv1(x1)
        v1 = self.conv2(x2)
        v2 = v1.flatten(2)
        qk = nn.functional.linear(v2, v2)
        v3 = qk.mul(self.scale_factor)
        v4 = nn.functional.softmax(v3, dim=-1)
        v4 = nn.functional.dropout(v4, 0.3) 
        output = nn.functional.linear(v4, v1.transpose(-2, -1).flatten(2))
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
