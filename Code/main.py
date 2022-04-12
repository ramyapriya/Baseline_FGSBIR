import torch
import os
import time
from utils import build_airobj, get_batch_objects, evaluate
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import yaml



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')
    parser.add_argument("--airobj_config_file", default='./config/train_airobj.yaml', type=str)
    parser.add_argument("--superpoint_config_file", default='./config/superpoint_inference.yaml', type=str)
    parser.add_argument("--seg_config_file", default='./config/maskrcnn_inference.yaml', type=str)
    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--root_dir', type=str, default='./../')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=1)

    hp = parser.parse_args()
    configs = {}

    # Read superpoint config file
    superpoint_config_file = hp.superpoint_config_file
    f = open(superpoint_config_file, 'r', encoding='utf-8')
    superpoint_configs = f.read()
    superpoint_configs = yaml.safe_load(superpoint_configs)
    configs['superpoint_configs'] = superpoint_configs

    # Read segmentation config file
    seg_config_file = hp.seg_config_file
    f = open(seg_config_file, 'r', encoding='utf-8')
    seg_configs = f.read()
    seg_configs = yaml.safe_load(seg_configs)
    configs['seg_configs'] = seg_configs

    # Read airobj config file
    airobj_config_file = hp.airobj_config_file
    f = open(airobj_config_file, 'r', encoding='utf-8')
    airobj_configs = f.read()
    airobj_configs = yaml.safe_load(airobj_configs)
    configs['airobj_configs'] = airobj_configs
    save_dir = airobj_configs['save_dir']

    print('Loaded all 3 configs!')
    
    # Dataloader
    dataloader_Train, dataloader_Test = get_dataloader(hp, configs, dataset_type='FGSBIR_AirObj')
    print(hp)

    print('Dataloader instantiated')

     # Load AirObject model
    model = build_airobj(airobj_configs)
    print('Loaded Airobject model')

    # Triplet loss
    criterion = torch.nn.TripletMarginLoss(margin=0.2)

    # Initializing trainable and non-trainable parameters
    train_parameters = []
    for name, p in model.named_parameters():
        if "tcn" in name:
            train_parameters.append(p)
        else:
            p.requires_grad = False
    optimizer = torch.optim.Adam(train_parameters, lr=hp.learning_rate)
    
    step_count, top1, top10 = -1, 0, 0
    
    # Train loop
    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            
            # Get Sequence of Points, Descs & Adjs grouped by Objects
            anchor_points, anchor_descs, anchor_adjs, iids_a = get_batch_objects(batch_data, device, 'sketch')
            positive_points, positive_descs, positive_adjs, iids_p = get_batch_objects(batch_data, device, 'positive')
            negative_points, negative_descs, negative_adjs, iids_n = get_batch_objects(batch_data, device, 'negative')

            
            invalid_ids = list(set(iids_a + iids_p + iids_n))

            fin_anchor_points, fin_anchor_descs, fin_anchor_adjs = [], [], []
            fin_positive_points, fin_positive_descs, fin_positive_adjs = [], [], []
            fin_negative_points, fin_negative_descs, fin_negative_adjs = [], [], []

            for idx in range(len(anchor_points)):
                if idx not in invalid_ids:
                    fin_anchor_points.append(anchor_points[idx])
                    fin_anchor_descs.append(anchor_descs[idx])
                    fin_anchor_adjs.append(anchor_adjs[idx])

                    fin_positive_points.append(positive_points[idx])
                    fin_positive_descs.append(positive_descs[idx])
                    fin_positive_adjs.append(positive_adjs[idx])

                    fin_negative_points.append(negative_points[idx])
                    fin_negative_descs.append(negative_descs[idx])
                    fin_negative_adjs.append(negative_adjs[idx])
                
            step_count = step_count + 1
            start = time.time()

            model.train()
            optimizer.zero_grad()
            sketch_img_desc = model(fin_anchor_points, fin_anchor_descs, fin_anchor_adjs) #invalid_ids)
            positive_img_desc = model(fin_positive_points, fin_positive_descs, fin_positive_adjs) #invalid_ids)
            negative_img_desc = model(fin_negative_points, fin_negative_descs, fin_negative_adjs) #invalid_ids)
            
            print(sketch_img_desc.shape[0], positive_img_desc.shape[0], negative_img_desc.shape[0])

        
            loss = criterion(sketch_img_desc, positive_img_desc, negative_img_desc)
            loss.backward()
            
            optimizer.step()

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss.item(), top1, top10, time.time()-start))

            if step_count % hp.eval_freq_iter == 0:
                save_path = os.path.join(save_dir, hp.dataset_name + '_model_e{}.pth'.format(i_epoch))
                with torch.no_grad():
                    top1_eval, top10_eval = evaluate(dataloader_Test, model)
                    print('results : ', top1_eval, ' / ', top10_eval)
                    torch.save(model.state_dict(), save_path)
                if top1_eval > top1:
                    save_path = os.path.join(save_dir, hp.dataset_name + '_model_best.pth')
                    torch.save(model.state_dict(), save_path)
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')
    