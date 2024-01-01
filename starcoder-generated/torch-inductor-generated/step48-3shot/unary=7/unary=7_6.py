
        module.relu = torch.nn.ReLU()
        self.conv  = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 8, 1, stride=1, padding=0)
 
        self.fc     = torch.nn.Linear(8*3*3, 8)
 
    def forward(self, x):
        if self.training:
            l1 = self.conv2(self.relu(self.conv(x)))
            l2 = l1 * torch.clamp(min=0, max=6, l1 + 3)
            l3 = l2 / 6
            x  = x + l3.reshape(-1, 8*3*3)
            x  = self.fc(x)
 
            return x
        else:
            x = self.fc(x.reshape(1,-1))
            return x

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
