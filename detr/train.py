import torch
from torch.utils.data import Dataset, DataLoader

import yaml, json, box
import util.misc as utils
from model import build_model
from engine import evaluate, train_one_epoch
from dataloader.dataset import CustomDataset

def train(args):
    device = torch.device(args.scheduler.device)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.scheduler.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.scheduler.lr,
                                  weight_decay=args.scheduler.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.lr_drop)
    
    if args.model.frozen_weights is not None:
        checkpoint = torch.load(args.model.frozen_weights, map_location='cpu')
        model.detr.load_state_dict(checkpoint['model'])
        
    if args.scheduler.resume:
        if args.scheduler.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.scheduler.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.scheduler.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.scheduler.start_epoch = checkpoint['epoch'] + 1

    ## start ##
    dataset = CustomDataset(data_root = DATA_ROOT, csv_file=csv_file, transform= t, infer=False)
    data_loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print("Start training")
    for epoch in range(args.scheduler.start_epoch, args.scheduler.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.scheduler.clip_max_norm)
        
        lr_scheduler.step()
        utils.save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, args.scheduler.save_path + 'checkpoint.pth')
    
if __name__ == "__main__":
    with open('./detr/config.yaml') as f:
        args = yaml.safe_load(f)
        args = box.Box(args)
    # train(args)
    print(args.model)
    