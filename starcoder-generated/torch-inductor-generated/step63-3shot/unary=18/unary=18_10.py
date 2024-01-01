
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x):
        # [batch_size, channels, H]
        v1 = x.relu()
        # [batch_size, channels, H-2]
        v2 = x[:, :, 1:-1].relu()
        # [batch_size, channels, H-4]
        v3 = x[:, :, 2:-2].relu()
        # [batch_size, channels, H-2]
        v4 = x[:, :, 1:-1].relu()
        # [batch_size, channels, H-4]
        v5 = x[:, :, 2:-2].relu()
        # [batch_size, channels, H-2]
        v6 = x[:, :, 1:-1].relu()
    	# [batch_size, 32, channels, H-2]
        v7 = torch.stack([v1, v2, v3], dim=2)
        # [batch_size, 16, channels, H-2]
        v8 = torch.stack([v4, v5, v6], dim=2)
        # [batch_size, 32, H-2]
        v9 = v7.mean(dim=3)
        # [batch_size, 16, H-2]
        v10 = v8.mean(dim=3)
        # [batch_size, channels, H-2]
        v11 = v9.relu()
        # [batch_size, 8, H-2]
        v12 = v10.relu()
        # [batch_size, 8, H-4]
        v13 = v11[:, :, 2:-2]
        # [batch_size, 16, H-4]
        v14 = v10[:, :, 2:-2]
        # [batch_size, 8, H-4]
        v15 = v12[:, :, :-2]
        # [batch_size, channels, H-4]
        v16 = torch.stack([
            v13.relu(),
            v14.relu(),
            v15.relu(),
            v11[:, :, :2].relu(),
            v12[:, :, 1:-1].relu(),
            v12[:, :, 2:-2].relu(),
        ], dim=2)
        # [batch_size, 8, H-2]
        v17 = v14.relu()
    	# [batch_size, 32, H-2]
        v18 = v16.mean(dim=2)
    	# [batch_size, 8, H-2]
        v19 = v17.relu()
        # [batch_size, 8, H-4]
        v20 = v13.relu()
        # [batch_size, 8, H-2]
        v21 = v12.relu()
        # [batch_size, 8, H-2]
        v22 = v21[:, :, 1:-1]
        # [batch_size, 16, H-2]
        v23 = v20[:, :, 1:-1]
        # [batch_size, 8, H-2]
        v24 = v21[:, :, :1]
        # [batch_size, channels, H-2]
        v25 = torch.stack([
            v23.relu(),
            v20[:, :, :2].relu(),
            v24.relu(),
            v22.relu(),
            v19[:, :, :-1].relu(),
            v11.relu(),
            v15.relu(),
            v19[:, :, 1:-1].relu(),
            v21[:, :, 2:-2].relu(),
        ], dim=2)
        # [batch_size, 8, channels, H-2]
        v26 = torch.stack([
            torch.stack([v25[i, :, j].relu() for i in range(v25.shape[0])], dim=0) for j in range(v25.shape[2])
        ], dim=2)
        v27 = v26.mean(dim=3)
        v28 = torch.sigmoid(v27)
        return v28
# Inputs to the model
x1 = torch.randn(1, 16, 4, 4)
