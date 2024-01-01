
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, stride=1, padding=0)
    def forward(self, x1, other1=1):
        v1 = self.conv(x1)
        v2 = F.relu(v1, True)
        v3 = v2 + 0.2
        v5 = my_fake_relu(v3, 0.1)
        v6 = v5 + 0.3
        v7 = v6 + 0.4
        v10 = my_fake_relu(v7, 0.5)
        v13 = v10 + 0.6
        v53 = v13 + 0.7
        v55 = my_fake_relu(v53, 0.8)
        v58 = v55 + 0.9
        v61 = v58 + 1.0
        v62 = my_fake_relu(v61, other1)
        v66 = v62 + other1
        v70 = v66 + 0.1
        v73 = v70 + 0.1
        v75 = my_fake_relu(v73, 0.1)
        v76 = v75 + 0.1
        v77 = my_fake_relu(v76, 0.1)
        v80 = my_fake_relu(v77, other1)
        v82 = my_fake_relu(v80, other1)
        v84 = v82 + 0.1
        v87 = v84 + 0.1
        v89 = my_fake_relu(v87, other1)
        v90 = v89 + 0.1
        v91 = my_fake_relu(v90, 0.1)
        v94 = my_fake_relu(v91, 0.1)
        v96 = my_fake_relu(v94, other1)
        v97 = v96 + other1
        v101 = my_fake_relu(v97, 0.1)
        v104 = v101 + 0.1
        v107 = v104 + 0.1
        v109 = my_fake_relu(v107, 0.1)
        v110 = my_fake_relu(v109, 0.1)
        v111 = my_fake_relu(v110, 0.1)
        v112 = my_fake_relu(v111, other1)
        v113 = my_fake_relu(v112, other1)
        result = my_fake_relu(v113, 0.1)
        return result
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
