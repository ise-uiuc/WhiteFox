
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(21, 6, 1, stride=1, padding=1)
    def forward(self, img0):
        var1 = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device='cuda')
        var2 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var3 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var4 = torch.zeros(1, 6, 3, 1, 1, 1, dtype=torch.float32, device='cuda')
        var5 = torch.zeros(1, 6, 3, 1, dtype=torch.float32, device='cuda')
        var6 = torch.zeros(1, 6, 3, 1, 1, 1, dtype=torch.float32, device='cuda')
        var7 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var8 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var9 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var10 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var11 = torch.zeros(1, 6, 1, 1, dtype=torch.float32, device='cuda')
        var12 = torch.tensor(24.0, dtype=torch.float32, device='cuda')
        conv1 = self.conv(img0)
        mul1 = conv1 * var2
        if conv1 > var1:
            # 18 ops
            if torch.isnan(conv1):
                # 2 ops
                var2 = torch.ones_like(var2)
            else:
                # 3 ops
                var3 = torch.zeros_like(var3)
            # 24 ops
            mul2 = conv1 * var3
            # 7 ops
            var4 = torch.flatten(var3, 2)
            # 13 ops
            var5 = torch.flatten(torch.reshape(var4, var6.size()), 0)
            # 7 ops
            if var5.isnan().any():
                var3 = torch.ones_like(var3)
            # 3 ops
            if torch.isnan(conv1):
                # 2 ops
                if var8.isnan().any():
                    # 4 ops
                    var5 = torch.flatten(torch.reshape(var4, var6.size()), 0)
                # 2 ops
                var5 = torch.flatten(torch.reshape(var4, (10, 1)), 0)
                # 2 ops
                # 13 ops
                var7 = conv1 * var7
                # 13 ops
            # 17 ops
            var9 += var4
        # 15 ops
        if torch.isnan(conv1):
            # 1 ops
            if torch.isnan(conv1):
                # 2 ops
            # 15 ops
            var10 = torch.sum((var6), dim=0, keepdim=False)

        # 18 ops
        if not torch.isnan(var9).any():
            # 17 ops
            if var7.isnan().any():
                # 4 ops
                var9 += var4
            # 7 ops
        # 32 ops
        if not torch.isnan(var8).any():
            # 7 ops
            var9 += var4
        # 17 ops
        var12 += var5

        # 17 ops
        loss = 1.0 - 1.0 + var10 + conv1 + var12

        return loss
# Inputs to the model
img0 = torch.randn(1, 21, 37, 31)
