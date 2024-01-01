
class Model_old(torch.nn.Module):
     def __init__(self): super().__init__()
     self.conv = torch.nn.Conv2d(2, 6, 3)
     self.fc1 = torch.nn.Linear(64, 64, bias=True)
     self.fc2 = torch.nn.Linear(64, 64, bias=True)
     def forward(self, x1, x2, y1, y2, z):
     v1 = self.conv(x2)
     v2 = self.conv(y2)
     v3 = self.fc1(z)
     v4 = self.conv(v1)
     v5 = self.conv(v3).add(torch.randn_like(v3.clone())).add(torch.randn_like(v3.clone()))
     v6 = v4 + y1 + x1
     v7 = v6 + v5 + 1
     v8 = v7 - 5
     v9 = self.fc2(z)
     return v8.add(v9)
