import torch
import time
from utils import build_airobj
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import yaml



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')
    parser.add_argument("--airobj_config_file", type = str)
    parser.add_argument("--superpoint_config_file", type = str)
    parser.add_argument("--seg_config_file", type = str)
    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=1)

    hp = parser.parse_args()
    
    # Read superpoint config file
    superpoint_config_file = hp.superpoint_config_file
    f = open(superpoint_config_file, 'r', encoding='utf-8')
    superpoint_configs = f.read()
    superpoint_configs = yaml.safe_load(superpoint_configs)
    hp['superpoint_configs'] = superpoint_configs
    
    # Read segmentation config file
    seg_config_file = hp.seg_config_file
    f = open(seg_config_file, 'r', encoding='utf-8')
    seg_configs = f.read()
    seg_configs = yaml.safe_load(seg_configs)
    hp['seg_configs'] = seg_configs
    
    # Read airobj config file
    airobj_config_file = hp.airobj_config_file
    f = open(airobj_config_file, 'r', encoding='utf-8')
    airobj_configs = f.read()
    airobj_configs = yaml.safe_load(airobj_configs)
    hp['airobj_configs'] = airobj_configs
    
    # Dataloader
    dataloader_Train, dataloader_Test = get_dataloader(hp, dataset_type='FGSBIR_AirObj')
    print(hp)

     # Load AirObject model
    model = build_airobj(airobj_configs)

    # Initializing trainable and non-trainable parameters
    train_parameters = []
    for name, p in model.named_parameters():
        if "tcn" in name:
            train_parameters.append(p)
        else:
            p.requires_grad = False

    step_count, top1, top10 = -1, 0, 0
    import pdb; pdb.set_trace()
    # Train loop
    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss, top1, top10, time.time()-start))

            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval = model.evaluate(dataloader_Test)
                    print('results : ', top1_eval, ' / ', top10_eval)

                if top1_eval > top1:
                    torch.save(model.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth')
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')