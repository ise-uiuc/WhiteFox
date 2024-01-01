
class Model(torch.nn.Module):
    def forward(self, x):
        t1 = F.conv2d(x, 3, 8, 1, stride=1, padding=1, bias=None) # Apply pointwise convolution with kernel size 1
        t2 = t1 + other # Add a tensor to the output of the convolution
        t3 = F.relu(t2) # Apply ReLU activation function
        return t3

