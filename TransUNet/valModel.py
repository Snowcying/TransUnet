from torchsummary import summary
from networks.vit_seg_modeling import VisionTransformer as ViT_seg

# net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

# summary(ViT_seg,(4,3,224,224))