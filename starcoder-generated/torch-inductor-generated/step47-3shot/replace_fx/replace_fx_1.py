
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
    def forward(self, x1, x2):
        t1 = torch.nn.functional.relu(x1)
        t2 = torch.nn.functional.max_pool2d(t1, 2)
        t3 = torch.nn.functional.relu(t2)
        t4 = torch.nn.functional.max_pool2d(t3, 1)
        t5 = torch.nn.functional.relu(t4)
        t6 = torch.nn.functional.max_pool2d(t5, 1)
        t7, _ = torch.max(t6, 1)
        t8 = torch.mean(t7, 1)
        t9 = t6 + torch.reshape(t8, (-1, 1, 1, 1024))
        t10 = torch.nn.functional.relu(t9)

        u1 = torch.nn.functional.relu(x2)
        u2 = torch.nn.functional.max_pool2d(u1, 2)
        u3 = torch.nn.functional.relu(u2)
        u4 = torch.nn.functional.max_pool2d(u3, 1)
        u5 = torch.nn.functional.relu(u4)
        u6 = torch.nn.functional.max_pool2d(u3, 1)
        u7, _ = torch.max(u6, 1)
        u8 = torch.mean(u7, 1)
        u9 = u6 + torch.reshape(u8, (-1, 1, 1, 1024))
        u10 = torch.nn.functional.relu(u9)
        return t10 + u10, t10 * u10, t10 - u10
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
