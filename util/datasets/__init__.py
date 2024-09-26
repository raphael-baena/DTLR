# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'synthetic_line_OCR_general':
        from .synthetic_lines_general import build_synthetic_line_OCR_general
        return build_synthetic_line_OCR_general(image_set, args)
    if args.dataset_file == 'google1000':
        from .google1000 import build_google1000
        return build_google1000(image_set, args)
    if args.dataset_file == 'IAM':
        from .IAM import build_iam
        return build_iam(image_set, args)
    if args.dataset_file == 'READ':
        from .READ import build_READ
        return build_READ(image_set, args)
    if args.dataset_file == 'RIMES':
        from .RIMES import build_RIMES
        return build_RIMES(image_set, args)
    # Ciphers
    if args.dataset_file =='borg':
        from .borg import build_borg
        return build_borg(image_set, args)
    if args.dataset_file == 'copiale':
        from .copiale import build_copiale
        return build_copiale(image_set, args)
    # Chinese
    if args.dataset_file =='HWDB_synth':
        from .HWDB_Synth import build_synthetic_HWDB
        return build_synthetic_HWDB(image_set, args)
    if args.dataset_file == 'HWDB':
        from .HWDB import build_HWDB
        return build_HWDB(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
