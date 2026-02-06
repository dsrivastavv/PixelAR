from dynamic_tokenization.dataset.imagenet import build_imagenet, build_imagenet_code
from dynamic_tokenization.dataset.coco import build_coco


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')