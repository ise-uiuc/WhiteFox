
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        t1 = torch.relu(x2)
        x1003 = self.conv3(x1)
        t2 = self.conv2(x2)
        t1001 = torch.sigmoid(self.conv4(t2) + x1003)
        t3 = t1 + t1001
        x1004 = self.conv2(t1)
        t4 = self.conv2(t3)
        x1005 = self.conv3(x1)
        t5 = self.conv2(t4)
        x1006 = torch.relu(x1005)
        t6 = self.conv2(x1004)
        x1007 = torch.relu(x1005)
        v_1001 = torch.exp(t6 + t5)
        t7 = v_1001 + x1007
        x1008 = torch.relu(x1006)
        t8 = torch.sigmoid(t8)
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.exp(v3)
        v5 = torch.relu(v4)
        v6 = torch.sigmoid(v5)
        v7 = v6 + v2
        v8 = v6 + v7
        v_1002 = torch.relu(v6)
        v9 = v_1002 + v2
        v10 = torch.sigmoid(v9)
        v11 = torch.sigmoid(v8)
        v12 = torch.sigmoid(v7)
        t1002 = v10 * v11
        t1003 = v7 * v12
        v13 = t1 + t1002 + t1003
        v14 = torch.sigmoid(v13)
        t1004 = torch.sigmoid(v14 + t1001)
        v15 = v2 + x2
        v16 = torch.tanh(v15)
        v17 = t1003 + v14
        v18 = t7 + v12
        v19 = v18 + v11
        v20 = v16 + t1001
        v21 = v10 + x2
        v22 = torch.sigmoid(v21)
        v23 = v17 + v19
        v24 = v23 + v11
        t1005 = v22 * torch.sigmoid(v13 + v16)
        x1002 = self.conv1(x1)
        t1006 = torch.sigmoid(v17)
        t1007 = torch.sigmoid(v16)
        t1008 = torch.sigmoid(v24)
        x1001 = v20 + v22
        v_1001 = torch.relu(v23)
        v_1002 = torch.sigmoid(x1001)
        v_1003 = torch.sigmoid(v20)
        x1003 = v23 + v24
        v25 = v24 + v20
        x1004 = v20 + x1003
        x1005 = v17 + v20
        x1006 = v16 + t1006
        v_1004 = torch.sigmoid(x1004)
        v_1005 = torch.sigmoid(x1005)
        t_1001 = v_1004 + v_1005
        v26 = v_1003 + t_1001
        v_1006 = torch.sigmoid(x1006)
        v_1007 = v_1006 + v_1005
        v27 = v_1007 + v_1004
        v_1008 = v_1004 + v_1007
        t_1002 = v_1008 * torch.relu(x1003)
        x1007 = self.conv2(x1003)
        t_1003 = v26 + v27
        t_1004 = torch.sigmoid(x1004)
        v_1009 = torch.tanh(x1007)
        v28 = v_1009 + x1006
        t_1005 = v_1009 + x1002
        t_1006 = v_1002 + v_1009
        t1009 = torch.sigmoid(self.conv4(v10 * v11) + x1007)
        v_1010 = torch.sigmoid(x1007)
        t1010 = torch.relu(v_1010 + t1009)
        t1011 = torch.sigmoid(torch.relu(v_1010 + t1009))
        t1012 = torch.sigmoid(v_1010 + t1011)
        t1013 = torch.sigmoid(torch.relu(v_1010))
        t1014 = t1013 + t1012
        v_1011 = torch.tanh(x1008)
        v29 = t1014 + v_1011
# Inputs to the model
x1 = torch.randn(1, 1, 64, 128)
