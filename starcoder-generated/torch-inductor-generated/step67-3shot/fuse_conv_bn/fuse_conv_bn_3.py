
model = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
model.eval()
def model_inference_with_images(self, images):
    images = list(image.to(self.device) for image in images)
    heights = [image.shape[-2] for image in images]
    widths = [image.shape[-1] for image in images]
    image = torch.stack(images, 0)

    x = self.backbone(image)

    x0_h, x0_w = x[0].shape[-2:]
    x1 = self.transpose_layer(x[1])
    x2 = self.transpose_layer(x[2])
    x3 = self.transpose_layer(x[3])
    x = [
        F.interpolate(x0, size=(x1.shape[-2], x1.shape[-1]), mode='bilinear', align_corners=False),
        x1,
        F.interpolate(x2, size=(x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False),
        x3
    ]
    if self.aux:
        x0_h, x0_w = x[2].shape[-2:]
        aux_out = self.aux_convs(torch.cat(x, 1))
        aux_out = self.aux_header(aux_out)

        aux_out = F.interpolate(aux_out, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
    x0_h, x0_w = x[0].shape[-2:]
    x1 = self.header0(torch.cat(x, 1))
    # comment out the following line to test error message when run with PyTorch 1.6.0 or 1.6.1, but not PyTorch 1.5.1
    x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
    x2 = torch.cat([x[2], x1], dim=1)
    x2 = self.header1(x2)
    x0_h, x0_w = x[0].shape[-2:]
    out = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
    if self.aux:
        out = [out, aux_out]
    return out
model.forward = model_inference_with_images.__get__(model, torchvision.models.segmentation.FCN)

x0 = torch.randn(2, 3, 224, 224)
x1 = torch.randn(2, 3, 224, 224)
x2 = torch.randn(2, 3, 224, 224)
x3 = torch.randn(2, 3, 224, 224)
x = [x0, x1, x2, x3]
model_torch = self.model(x)
# Inputs to the model
x0 = torch.randn(1, 3, 224, 224)
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
x3 = torch.randn(1, 3, 224, 224)
x = [x0, x1, x2, x3]
