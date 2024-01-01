
for cnn_module in [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]:
    t_input = torch.tensor(1., requires_grad=True)
    for N in range(10):
        x_input = torch.sum(t_input).data
        x_input.requires_grad = True
        for C in range(10):
            y_input = x_input.view(1, -1)
            y_input.requires_grad = True
            z_input = y_input + torch.randn(C, 1) 
            z_input.requires_grad = True
            _ = cnn_module(N+C, C)
            t__ = torch.cat([z_input]*N, dim=0)
            ___ = t__[:, 0]
            