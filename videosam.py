import argparse
import os
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision import transforms
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import random
from torch.optim import SGD,Adam
from torchvision import transforms
import webdataset as wds
from memory import SelfSupervisedMemory




unloader = transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def show_mask(mask, ax, color, random_color=False):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    #     color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=4))
    # ax.text(x0, y0, label)


def match_from_embds(tgt_embds, cur_embds,mem):
    for len_cur in range(cur_embds.shape[0]):
        if cur_embds.sum(dim=1)[len_cur]==0:
            break
    for len_tgt in range(tgt_embds.shape[0]):
        if tgt_embds.sum(dim=1)[len_tgt]==0:
            break
        
    #cur_valid_idx = mem.remove_duplicated_slot_id(cur_embds)
    #tgt_valid_idx = mem.remove_duplicated_slot_id(tgt_embds)

    cur_emb = cur_embds[0:len_cur] / cur_embds[0:len_cur].norm(dim=1)[:, None]
    tgt_emb = tgt_embds[0:len_tgt] / tgt_embds[0:len_tgt].norm(dim=1)[:, None]
    cos_sim = torch.mm(cur_emb, tgt_emb.transpose(0, 1))

    cost_embd = 1 - cos_sim

    C = 1.0 * cost_embd
    print(C.shape)
    indices = torch.argmin(C.transpose(0, 1), dim=-1)
    indices_all=torch.zeros([mem.object_num])
    indices_all[0:len_cur]=indices

    # C = C.cpu()
    # print(C.transpose(0, 1))
    #
    # indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
    # print(indices)
    # indices = indices[1]  # permutation that makes current aligns to target
    return indices_all

def rollout_loss(segmentations,masks,smooth):
        # only calucate frames [4:] loss
        # inputs = F.sigmoid(segmentations[:, 2:])

        h,w = segmentations.shape
        inputs = segmentations.reshape(-1, h, w)
        targets = masks.reshape(-1, h, w)
        ce_loss = F.binary_cross_entropy(inputs, targets.float())

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1).float()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1-dice
        total_loss = ce_loss + dice_loss
        return total_loss

def EM_loss_ori(pred_feature, tgt_feature,attn_index):
    #rec_tgt = rec_tgt.reshape(-1,3,h,w).unsqueeze(1).unsqueeze(2).repeat(1,n_slots,n_buffer,1,1,1)
    #reconstructions = reconstructions.reshape(-1, n_buffer, 3, h, w).unsqueeze(1).repeat(1,n_slots,1,1,1,1)
    #rec_pred = reconstructions * masks_vis
    #rec_tgt_ = rec_tgt * masks_vis
    #loss = torch.sum(F.binary_cross_entropy(segmentations, masks.float(), reduction = 'none'), (-1,-2)) / (h*w) + 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3,-2,-1))
        # be = torch.sum(F.binary_cross_entropy(segmentations, masks.float(), reduction = 'none'), (-1,-2)) / (h*w)
        # rec = 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3,-2,-1))
        # print(be[0,1,1], rec[0,1,1])
    loss_weight=1
    b=attn_index.shape[0]
    n_obj=tgt_feature.shape[1]
    n_buffer=pred_feature.shape[1]
    loss=torch.sum(F.mse_loss(pred_feature, tgt_feature, reduction="none"), (-3,-2,-1))
    #print("mse_loss")
    #print(loss)
    total_loss = torch.sum(attn_index * loss, (0,1,2)) / (b  * n_obj * n_buffer)
    return (total_loss) * loss_weight
    #return attn_index.sum()

def EM_loss(cur_feature,pred_feature,attn_index):
    b=attn_index.shape[0]
    n_obj=cur_feature.shape[1]
    n_buffer=pred_feature.shape[1]
    loss=torch.tensor([0]).to(float).to(device)
    cur_sum=cur_feature.sum(dim=-1)
    pred_sum=pred_feature.sum(dim=-1)

    for n in range(b):
        cur_empty_idx = cur_sum[n,:].nonzero(as_tuple=True)[0]
        pred_empty_idx = pred_sum[n,:].nonzero(as_tuple=True)[0]
        for i in range(n_obj):
            for j in range(n_buffer):
                #if tgt_feature[n,i,:].sum().to(torch.int)!=0 & pred_feature[n,j,:].sum().to(torch.int)!=0:
                index_prob = attn_index[n,i,j].to(device)   
                loss+=index_prob*F.mse_loss(cur_feature[n,i,:], pred_feature[n,j,:], reduction="sum")
    loss= loss/ (b  * n_obj)
    return loss


def visualization_gif(mem,indices,image_show,masks_show,boxes_filt_show, frame_id,output_dir):
    color_list_new=[]
    color_list_all = [np.array([1,1,1,0.6]),np.array([0,1,0,0.6]),np.array([1,0,0,0.6]),
                      np.array([0,1,1,0.6]),np.array([0,0,1,0.6]),np.array([1,0,1,0.6]),
                      np.array([1,1,0,0.6]),np.array([0.5,0.5,0,0.6]),np.array([0,0.5,0.5,0.6]),np.array([0.5,0,0.5,0.6])]
    if frame_id==0:
        color_list_new = color_list_all
    if frame_id>0:
        #print(indices)
        for i in range(len(indices)):
            color_list_new.append(color_list_all[indices[i]])
    plt.figure(figsize=(10, 10))
    plt.imshow(image_show)
    j=0
    #print(boxes_filt_show.shape)
    if masks_show!= None: 
        for i, mask in enumerate(masks_show):
            #print(i)

            color = color_list_new[i]
            #print(color)
            show_mask(mask.cpu().numpy(), plt.gca(), color, random_color=False)
        old_mem_idx = (boxes_filt_show.sum(dim=1)!= 0).nonzero(as_tuple=True)[0].tolist()
        for id in old_mem_idx:
            #print("boxes",j,"\n")
            show_box(boxes_filt_show[id,:].cpu().numpy(), plt.gca())

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "frame%05d.jpg" % frame_id),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    # image = Image.open(os.path.join(output_dir, "frame%05d.jpg" % frame_id))
    image = cv2.imread(os.path.join(output_dir, "frame%05d.jpg" % frame_id))
    
    return image

def non_dup_match(attn_index,cur_masks,prev_masks,mem):
    batch,n_obj,n_buffer=attn_index.shape
    indices_all=torch.zeros([batch,n_obj])
    for b in range(batch):
        
        cur_valid_idx = mem.remove_duplicated_slot_id(cur_masks[b])       
        prev_valid_idx = mem.remove_duplicated_slot_id(prev_masks[b]) 
        n_cur=len(cur_valid_idx)
        n_prev=len(prev_valid_idx)
        attn_index_vis=attn_index[b,:n_cur,:n_prev].clone()
        match=torch.zeros([n_cur])
        indices=torch.zeros([n_cur])-1
        #tmp= range(n_cur)
        if b==0:
            print(n_cur)
            print(n_prev)
        for i in range(n_cur):
            indice=torch.argmax(attn_index_vis[i,:n_prev],dim=-1)
            #tmp.remove(indice)
            indices[i]=indice
            attn_index_vis[:,indice]=-torch.inf
        indices_all[b,0:n_cur]=indices
        if n_cur>n_prev:
            for j in range(n_cur-n_prev):
                indices_all[b,j+n_prev]=j+n_prev
    indices_all=indices_all.to(torch.int)
    return indices_all

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")

    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")

    parser.add_argument("--train_data_path",type=str,default="./filter_data/shard-{000000..002399}.tar")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_objects', type=int, default=10)
    parser.add_argument('--memory_len', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=400)
    parser.add_argument('--log_path', type=str, default="/home/ubuntu/exp/mem-videosam-main/log")
    parser.add_argument('--output_path', type=str, default="/home/ubuntu/exp/mem-videosam-main/output")
    parser.add_argument('--video_len', type=int, default=8)
    parser.add_argument("--lr_roll", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)

    args = parser.parse_args()
    device=args.device
    vis_batch=0
    output_dir=args.output_path

    #import memory module
    torch.multiprocessing.set_start_method('spawn')

    mem=SelfSupervisedMemory(embed_dim=args.embed_dim,num_objects=args.num_objects,memory_len=args.memory_len)
    mem=mem.cuda()
    #mem= nn.DataParallel(mem)
    batch_size=args.batch_size

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device
    lr= args.lr

    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)
    #optimizer = Adam([{'params':mem.roll_out_module.parameters(),'lr':lr*10},
    #                    {'params':mem.MultiHead_1.parameters(),'lr':lr},
    #                    {'params':mem.MultiHead_2.parameters(),'lr':lr}])
    optimizer = Adam(mem.parameters(), lr = 0.005)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    video_lenth = args.video_len
    num_buffer = args.num_objects


    train_dataset = wds.WebDataset(args.train_data_path,shardshuffle=True).shuffle(100).decode("rgb")
    loader = wds.WebLoader(train_dataset, num_workers=1, batch_size=batch_size)
    vis_batchlen=20
    mem.train()

    #loss_list_all=torch.zeros([args.epoch_num*300],requires_grad=False)
    #loss_list_all_pred=torch.zeros([args.epoch_num*300],requires_grad=False)
    for epoch in range(args.epoch_num):
        print("=================> Epoch:  ", epoch)
        
        loss_list=torch.zeros([batch_size,video_lenth-1]).cuda()
        
        for batch,batch_data in enumerate(loader):
            print("=================> Batch:  ", batch)
            gif = []
            loss_list_EM=torch.zeros([video_lenth-1]).cuda()
            loss_list_EM_pred=torch.zeros([video_lenth-1]).cuda()
            mask_all_prev=torch.zeros([batch_size,args.num_objects,256,256]).cuda()
            for frame_id in range(video_lenth):
                print("*************Processing batch ",batch, "frame ", frame_id)
                video_id=0
                observation_tokens=torch.zeros([batch_size,args.num_objects,args.embed_dim]).cuda()
                mask_all=torch.zeros([batch_size,args.num_objects,256,256]).cuda()
                
                for bs in range(batch_size):                    
                    object_tokens=batch_data[f'{frame_id:06d}object_tokens.pyd'][bs,:,:].squeeze()
          
                        
                    observation_tokens[bs,:,:]=object_tokens
                    non_emp_idx = (object_tokens.sum(dim=1)!= 0).nonzero(as_tuple=True)[0].tolist()
                    emp_idx = (object_tokens.sum(dim=1)== 0).nonzero(as_tuple=True)[0].tolist()
                    if len(non_emp_idx)>0:
                        if len(emp_idx)>0:
                            for i in emp_idx:
                                given_num=0
                                if len(non_emp_idx)>1:
                                    given_num=random.randint(0,len(non_emp_idx)-1)
                                observation_tokens[bs,i,:]=object_tokens[given_num,:]
                        observation_tokens[bs,:,:] = observation_tokens[bs,:,:] / observation_tokens[bs,:,:].norm(dim=1)[:, None]

                # draw output image
                
                mask_all=batch_data[f'{frame_id:06d}mask.pyd'].cuda()
                if frame_id==0:
                    mask_all_prev=mask_all
                else:
                    frame_prev=frame_id-1
                    mask_all_prev=batch_data[f'{frame_prev:06d}mask.pyd'].cuda()

                pred,attn_index,mem_buffers=mem(observation_tokens,mask_all,mask_all_prev,frame_id)
                mask_all_prev=mask_all
                #pred= pred/ pred.norm(dim=1)[:, None]
                for b in range(mem_buffers.shape[0]):
                    old_mem_idx = (mem_buffers[b].sum(dim=1)!= 0).nonzero(as_tuple=True)[0].tolist()
                    emp_mem_idx = (mem_buffers[b].sum(dim=1)== 0).nonzero(as_tuple=True)[0].tolist()
                    if len(emp_mem_idx)>0:
                        for i in emp_mem_idx:
                            given_num=0
                            if len(old_mem_idx)>1:
                                given_num=random.randint(0,len(old_mem_idx)-1)
                            mem_buffers[b,i] = mem_buffers[b,given_num]

                #visualization
                if batch%vis_batchlen==0:
                        
                    masks_show=batch_data[f'{vis_batch:06d}mask.pyd'][bs,:,:,:].squeeze()
                    boxes_filt_show=batch_data[f'{vis_batch:06d}box.pyd'][bs,:,:].squeeze() 
                    
                    image_show=batch_data[f'{vis_batch:06d}input.pyd'][bs,:,:,:].squeeze()

                    indices=[]
                    if frame_id>0:
                        print("attn_index")
                        print(attn_index[vis_batch,:,:])
                        attn_index_vis=attn_index[vis_batch,:,:]

                        prev_frame=frame_id-1
                        vis_obj=batch_data[f'{prev_frame:06d}object_tokens.pyd'][vis_batch,:,:].squeeze()
                        prev_num=7
                        if vis_obj.shape[0]>0:

                            non_emp_idx = (vis_obj.sum(dim=1)!= 0).nonzero(as_tuple=True)[0].tolist()
                            prev_num=len(non_emp_idx)
                        indices=torch.argmax(attn_index_vis[:,0:prev_num], dim=-1)

                        print(indices)
                    img=visualization_gif(mem,indices,image_show,masks_show,boxes_filt_show,frame_id,output_dir)
                    try:
                        img = Image.fromarray(img, mode='RGB')
                        gif.append(img)
                    except:
                        pass

                
                if frame_id>=1:
                    #pred_mask=mask_decoder(sam.mask_decoder.output_hypernetworks_mlps[0],pred,upscaled_embedding_all,object_num,predictor).cuda()
                    #mask_all=mask_all.float()#mask_all 0,1 type            
                    #loss_value_EM=EM_loss(observation_tokens,pred,attn_index)
                    loss_value_EM=EM_loss(observation_tokens,mem_buffers,attn_index)
                    loss_value_EM_pred=EM_loss(observation_tokens,pred,attn_index)
                    loss_list_EM[frame_id-1]=loss_value_EM+loss_value_EM_pred
                    loss_list_EM_pred[frame_id-1]=loss_value_EM_pred
                            
                    if frame_id>video_lenth:
                        break

            
            loss=loss_list_EM.mean()
            loss_pred=loss_list_EM_pred.mean()
            print("loss")
            print(loss)
            #print(loss_pred)
            #loss_list_all[epoch*300+batch]=loss
            #loss_list_all_pred[epoch*300+batch]=loss_pred
            #print(loss_list_all[0:(epoch*300+batch+1)])
            #print(loss_list_all_pred[0:(epoch*300+batch+1)])
            #print(loss_list_all[0:(epoch*300+batch+1)]-loss_list_all_pred[0:(epoch*300+batch+1)])
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            #for name, parms in mem.named_parameters():	
                #if name[0:9] == "MultiHead":
                    #print('-->name:', name)
                #    print('-->para:', parms)
            #        print('-->grad_requirs:',parms.requires_grad)
                    #print('-->grad_value:',parms.grad)

            global_step=epoch + batch
            writer.add_scalar('TRAIN/loss', loss.item(), global_step)
            writer.add_scalar('TRAIN/loss_buffer', loss.item()-loss_pred.item(), global_step)
            writer.add_scalar('TRAIN/loss_pred', loss_pred.item(), global_step)
            
            if batch%vis_batchlen==0:
                #save the gif
                os.makedirs("gif0", exist_ok=True)
                frame_one = gif[0]
                frame_one.save(f"./gif/videosam_{batch}_{epoch}.gif", format="GIF", append_images=gif,
                save_all=True, duration=100, loop=0)

            if batch%40==0:
                checkpoint = {
                    'model': mem.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                torch.save(checkpoint, os.path.join(args.log_path, 'checkpoint_videosam.pt.tar'))

    writer.close()
  
