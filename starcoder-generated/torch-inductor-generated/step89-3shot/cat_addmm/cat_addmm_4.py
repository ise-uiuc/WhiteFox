
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 2),
            nn.Linear(2, 2)
        )
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.cat((x, x, x), dim=-1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
# Model Ends

y = torch.randn(6, 2, 3) # Expect a tensor of shape (6, 3)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv1d(2, 3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape((30, 3))
        x = torch.mean(x, dim=0)
        return x

# Inputs to the model
x = torch.randn(6, 2, 2) # Shape of (batch_size, num_channel, length)

y = torch.randn(6, 2, 2, 3) # Expect a tensor of shape (6, 3)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(2, 3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape((x.size(0), -1))
        x = torch.mean(x, dim=1)
        return x

# Inputs to the model
x = torch.randn(6, 2, 2, 2) # Shape of (batch_size, num_channel, height, width)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(2, 3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        return x

# Inputs to the model
x = torch.randn(6, 2, 2, 2) # Shape of (batch_size, num_channel, height, width)

y = torch.randn(6, 5, 40, 40) # Expect a tensor of shape (6, 32000)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(2, 32, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        return x

# Inputs to the model
x = torch.randn(6, 2, 20, 20) # Shape of (batch_size, num_channel, height, width)

y = torch.randn(6, 5, 30) # Expect a tensor of shape (6, 150)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(2, 5, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape((x.size(0), -1))
        x = torch.mean(x, dim=1)
        return x

# Input to the model
x = torch.randn(6, 2, 20, 20) # Shape of (batch_size, num_channel, height, width)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.reshape((6, 1, -1, 7))
        x = x - x.mean()
        x = x.pow(2)
        x = x.mean(dim=1)
        x = x.reshape((6, -1))
        x = x.add(1).sqrt()
        x = x.div(7)
        return x

# Input to the model
x = torch.randn(6, 8, 2) # 6 batch, 8 in_features, 2 out_features


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(2, 3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size()[0] * x.size()[1], x.size()[-2], x.size()[-1])
        return x

# Input to the model
x = torch.randn(6, 2, 2, 2)  # Shape of (batch_size, num_channel, height, width)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(2, 3)
    def forward(self, x):
        x = x.view(12)
        x = self.layers1(x)
        return x

# Input to the model
x = torch.randn(4, 3, 2)  # Shape of (batch_size, num_channel, input dim)

