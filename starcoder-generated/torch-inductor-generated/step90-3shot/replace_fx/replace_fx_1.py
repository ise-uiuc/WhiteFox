
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Conv2d(2, 10, kernel_size=(5, 5))
    def forward(self, x):
        x1 = F.sigmoid(x)
        x2 = F.threshold(x, 1, 1, 1) 
        x3 = F.interpolate(x, size=(x1.size(-1), x1.size(-2))) 
        x4 = F.softmax(x)
        x5 = torch.softmax(x, 0) 
        x6 = F.softplus(x)
        x7 = F.elu(x)
        x8 = F.selu(x)
        x9 = F.silu(x)
        x10 = F.relu(x) 
        x11 = F.leaky_relu(x, negative_slope=0.01)
        x12 = F.logsigmoid(x)
        x13 = F.logsoftmax(x)
        x14 = F.tanh(x)
        x15 = F.hardtanh(x, min_val=-1.0, max_val=1.0) 
        x16 = F.sigmoid(x)
        x17 = torch.sigmoid(x)

        x18 = F.dropout(x, p=0.5)
        x19 = torch.dropout(x18, p=0.2) 
        x20 = F.dropout2d(x, p=0.5)
        x21 = F.dropout3d(x, p=0.5)
        x22 = F.alpha_dropout(x, p=0.5)

        x23 = torch.sigmoid(x) 
        x24 = F.sigmoid(x + 10)
        x25 = F.sigmoid(x18)
        x26 = F.sigmoid(x19 + 10)

        x27 = F.conv1d(x, x)
        x28 = F.conv2d(x, x, stride=2)
        x29 = F.conv3d(x, weight=x17, stride=[1]) 
        
        x30 = F.conv_transpose1d(x, x, stride=2)
        x31 = F.conv_transpose1d(x2, F.sigmoid(x16))
        x32 = F.conv_transpose3d(x, x, stride=(1,))

        x33 = F.linear(x, self.f1.weight)
        x34 = F.linear(x + 10, x27.permute(1, 0))

        x35 = torch.randn(1, 2, 1, 2)
        x36 = torch.randn(1, 2, 1, 2).to(dtype=torch.double) 
        x37 = torch.randn(1, 2, 2) 
        x38 = torch.randn(1, 2, 1, 2, device=torch.device("cuda"))

        x39 = (x2, x7)

        x40 = torch.cat([x39[0]] + [x39[1]], 0) 

        x41 = torch.rand_like(x) 

        x42 = x * 10
        x43 = x30 * torch.softmax(x)

        x44 = F.gelu(x)
        x45 = F.gelu(x2)
        x46 = F.gelu(x4)

        x47 = F.conv_transpose1d(x2, torch.clamp(x25, 0.5))
        x48 = F.conv_transpose1d(x2, torch.clip(x25, 0.5))

        y1 = x41 * x42 * x43 * x44 * x45 * x46
        return y1 
# Inputs to the model
x = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
