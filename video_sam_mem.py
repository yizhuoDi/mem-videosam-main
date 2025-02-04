import argparse
import os
from scipy.optimize import linear_sum_assignment
import torch
import pickle
from PIL import Image, ImageDraw, ImageFont
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import load_image, predict, annotate
import numpy as np
# segment anything
from segment_anything import build_sam, SamPredictor
from segment_anything.modeling import MaskDecoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

from torch.optim import SGD,Adam
from torchvision import transforms
import webdataset as wds
from memory import SelfSupervisedMemory
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from torchvision import transforms

transform = T.Compose(
    [
        T.RandomResize([256], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

unloader = transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def visualize(image, recon_dvae, recon_tf, attns, N=8):

    # tile
    tiles = torch.cat((
        image[:N, None, :, :, :],
        recon_dvae[:N, None, :, :, :],
        recon_tf[:N, None, :, :, :],
        attns[:N, :, :, :, :]
    ), dim=1).flatten(end_dim=1)

    # grid
    grid = vutils.make_grid(tiles, nrow=(1 + 1 + 1 + args.num_slots), pad_value=0.8)

    return grid


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=False, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, color, random_color=False):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    #     color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=4))
    # ax.text(x0, y0, label)

def match_from_mem(prev_num, object_tokens, memory, memory_table,frame_id, occupied_num, color_list, history_length = 10):
    color_dict = {}
    num_object = object_tokens.shape[0]
    print("num_object\n")
    print(num_object)
    if occupied_num == num_object:
        print("match")
        prev_tokens = memory[frame_id - 2, :occupied_num].reshape(-1, 256)
    else:
        if frame_id <= history_length:
            prev_tokens = memory[:frame_id-1, :occupied_num].reshape(-1, 256) # [L*occupied_num, 256]
        else:
            prev_tokens = memory[frame_id-1-history_length: frame_id-1, :occupied_num].reshape(-1, 256)
    #print(prev_tokens)
    prev_embds = prev_tokens / prev_tokens.norm(dim=1)[:, None]
    curr_embds = object_tokens / object_tokens.norm(dim=1)[:, None]
    cos_sim = torch.mm(prev_embds, curr_embds.transpose(0, 1))
    C = (1 - cos_sim) * 1.0
    # indices = torch.argmin(C.transpose(0, 1), dim=-1)
    if prev_tokens.shape[0] < num_object:
        indices = torch.argmin(C.transpose(0, 1), dim=-1)
        indices_obj_to_buffer = indices % occupied_num
    else:
        C = C.cpu()
        #C=torch.where(torch.isnan(C ), torch.full_like(C, 0), C)
        #print(C)
        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target
        # associate buffer_id to object_id
        indices_obj_to_buffer = torch.from_numpy(indices % occupied_num).to(device)
    #print(indices_obj_to_buffer, num_object)

    # find unmatched buffer_id
    unmatched_buffer_ids = []
    for buffer_id in range(occupied_num):
        if buffer_id not in indices_obj_to_buffer:
            unmatched_buffer_ids.append(buffer_id)
    # consider object-in, find the duplicate buffer_id
    out, count = torch.unique(indices_obj_to_buffer, return_counts=True)
    duplicate_buffer_id_index = (count > 1).nonzero(as_tuple=True)[0].tolist()
    unmatched_object_ids = []
    not_open_new_buffer = True
    if len(duplicate_buffer_id_index) > 0:
        duplicate_buffer_ids = out[duplicate_buffer_id_index]
        for duplicate_id in duplicate_buffer_ids:
            object_ids = (indices_obj_to_buffer == duplicate_id).nonzero(as_tuple=True)[0].tolist()
            inds_to_be_compared = indices[object_ids]
            cost_compare = []
            for i in range(inds_to_be_compared.shape[0]):
                cost_compare.append(C[inds_to_be_compared[i], object_ids[i]])
            # only keep the object with the largest similarity, other objects are considered as new objects
            matched_object_id = object_ids[cost_compare.index(min(cost_compare))]
            object_ids.remove(matched_object_id)
            for i in object_ids:
                unmatched_object_ids.append(i)

        # unmatched_object_ids are the new objects, activate new buffers for them
        num_new_objects = len(unmatched_object_ids)
        if occupied_num+num_new_objects <= num_buffer and num_object > prev_num:
            memory[frame_id-1, occupied_num: occupied_num+num_new_objects] = object_tokens[unmatched_object_ids]
            memory_table[occupied_num: occupied_num + num_new_objects] += 1
            for (idx, i) in enumerate(unmatched_object_ids):
                color_dict[str(i)]= color_list[occupied_num+idx]
            occupied_num += num_new_objects
            not_open_new_buffer = False
        else:
            not_open_new_buffer = True

    for object_id in range(num_object):
        if object_id not in unmatched_object_ids:
            buffer_id = indices_obj_to_buffer[object_id]
            memory[frame_id-1, buffer_id] = object_tokens[object_id]
            memory_table[buffer_id] = 1
            color_dict[str(object_id)]= color_list[buffer_id]
        else:
            if not_open_new_buffer:
                color_dict[str(object_id)] = color_list[-1]
    return memory, occupied_num, color_dict

def match_from_embds(tgt_embds, cur_embds):

    cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
    tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
    cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))

    cost_embd = 1 - cos_sim

    C = 1.0 * cost_embd

    indices = torch.argmin(C.transpose(0, 1), dim=-1)
    # C = C.cpu()
    # print(C.transpose(0, 1))
    #
    # indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
    # print(indices)
    # indices = indices[1]  # permutation that makes current aligns to target
    return indices

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

def mask_decoder(func,object_tokens,upscaled_embedding_all,object_num,predictor):
    #src = src.transpose(1, 2).view(b, c, h, w)
    #upscaled_embedding = mem.output_upscaling(src)
    #hyper_in_list: List[torch.Tensor] = []
    #hyper_in_list.append(func(object_tokens))
    #hyper_in = torch.stack(hyper_in_list, dim=1)
    #print(object_num)
    de_masks_all=torch.zeros([batch_size,256,256])
    for bs in range(batch_size):

        #print("masks.shape")
        _,b, c, h, w = upscaled_embedding_all.shape
        b=int(object_num[bs])
        if b>0:
            #print(object_tokens[bs].shape)
            hyper=func(object_tokens[bs,0:b,:])
            hyper=torch.unsqueeze(hyper,dim=1)
            #print(object_tokens[bs,b+1,:])
            #print(object_tokens[bs].sum(dim=1))
            res_masks = (hyper @ upscaled_embedding_all[bs,0:b,:,:,:].view(b, c, h * w)).view(b, -1, h, w)
            
            masks = predictor.model.postprocess_masks(res_masks, predictor.input_size, predictor.original_size)
            #print(predictor.model.mask_threshold)
            #masks_bool = masks > predictor.model.mask_threshold
            #one=torch.ones_like(masks)
            zero=torch.zeros_like(masks)
            #masks=torch.gt(masks,0)
            masks=torch.maximum(masks,zero)
            #masks=torch.where(masks > predictor.model.mask_threshold,one,zero)
            #print(masks.sum())
            #print(masks_bool.sum())
            #print(masks.sum())
            de_masks_all[bs,:,:]=masks.sum(dim=0).squeeze()
    #masks=func(object_tokens)
        
            #print(masks.shape)
    #masks = predictor.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
    one=torch.ones_like(de_masks_all)
    de_masks_all=torch.minimum(de_masks_all*100,one)
    return de_masks_all

def EM_loss(pred_feature, tgt_feature,attn_index):
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
    total_loss = torch.sum(attn_index * loss, (0,1,2)) / (b  * n_obj * n_buffer)
    return (total_loss) * loss_weight

def visualization_gif(mem,indices,image_show,masks_show,boxes_filt_show, pred_phrases_show,frame_id,output_dir):
    color_list_new=[]
    color_list_all = [np.array([1,1,1,0.6]),np.array([0,1,0,0.6]),np.array([1,0,0,0.6]),np.array([0,1,1,0.6]),np.array([0,0,1,0.6]),np.array([1,0,0,0.6])]
    color_list_new = color_list_all
    if frame_id>0:
        print(indices)
        for i in range(len(indices)):
            color_list_new.append(color_list_all[indices[i]])
    plt.figure(figsize=(10, 10))
    plt.imshow(image_show)
    j=0
    #print(boxes_filt_show.shape)
    for i, mask in enumerate(masks_show):
        print(i)

        color = color_list_new[i]
        print(color)
        show_mask(mask.cpu().numpy(), plt.gca(), color, random_color=False)
    for box, label in zip(boxes_filt_show, pred_phrases_show):
        #print("boxes",j,"\n")
        show_box(box.cpu().numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "frame%05d.jpg" % frame_id),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    # image = Image.open(os.path.join(output_dir, "frame%05d.jpg" % frame_id))
    image = cv2.imread(os.path.join(output_dir, "frame%05d.jpg" % frame_id))
    
    return image,color_list_new
                
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

    parser.add_argument("--train_data_path",type=str,default="./data/collision/shard-{000000..000007}.tar")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_objects', type=int, default=6)
    parser.add_argument('--memory_len', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=400)
    parser.add_argument('--log_path', type=str, default="/home/ubuntu/exp/mem-videosam-main/log")
    parser.add_argument('--output_path', type=str, default="/home/ubuntu/exp/mem-videosam-main/output")
    parser.add_argument('--video_len', type=int, default=8)
    parser.add_argument("--lr_roll", type=float, default=0.1)

    args = parser.parse_args()
    device=args.device
    vis_batch=2
    output_dir=args.output_path

    #import memory module
    mem=SelfSupervisedMemory(embed_dim=args.embed_dim,num_objects=args.num_objects,memory_len=args.memory_len)
    mem=mem.to(device)
    batch_size=8

    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)
    optimizer = Adam(mem.parameters(), lr = 0.001)  
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

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    video_lenth = args.video_len
    num_buffer = mem.object_num

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    #mask_decoder= sam.mask_decoder

    color_list = [np.concatenate([np.random.random(3), np.array([0.6])], axis=0) for _ in range(num_buffer)]

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    num_occupied = 0
    color_dict = {}

    train_dataset = wds.WebDataset(args.train_data_path,shardshuffle=True).shuffle(100).decode("rgb")
    loader = wds.WebLoader(train_dataset, num_workers=1, batch_size=batch_size)
    video_id=0
    mem.train()

    loss_list_all=torch.zeros([args.epoch_num],requires_grad=False)
    num=0
    for epoch in range(args.epoch_num):
        print("=================> Epoch:  ", epoch)
        mem.train()
        loss_list=torch.zeros([batch_size,video_lenth-3]).to(device)
        loss_list_EM=torch.zeros([video_lenth-3]).to(device)
        
        for batch,batch_img in enumerate(loader):
            gif = []
            for frame_id in range(video_lenth):
                video_id=0
                observation_tokens=torch.zeros([batch_size,mem.object_num,mem.embed_dim],requires_grad=False).to(device)
                mask_all=torch.zeros([batch_size,256,256]).to(device)
                upscaled_embedding_all=torch.zeros([batch_size,10,32,256,256]).to(device)
                object_num=torch.zeros([batch_size])
                for bs in range(batch_size):
                    print("*************Processing video ", video_id, "frame ", frame_id)
                    # run grounding dino model
                    image_input=batch_img[f'{frame_id:06d}.input.png'][bs,:,:,:].squeeze()
                    image=np.transpose(image_input,(2,0,1))
                    image=tensor_to_PIL(image)
                    image, _ = transform(image, None)
                    
                    #image.
                    boxes_filt, pred_phrases = get_grounding_output(
                        model, image, text_prompt, box_threshold, text_threshold, device=device
                    )

                    input_tensor=image.to(torch.device('cpu')).numpy()
                    in_arr=np.transpose(input_tensor,(1,2,0))
                    image = cv2.cvtColor(np.uint8(in_arr*255), cv2.COLOR_BGR2RGB)
                    predictor.set_image(image)
                    size = image.shape
                    H, W = size[1], size[0]
                    #print(boxes_filt.size(0))
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cuda()
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
                    if transformed_boxes.shape[0] == 0:
                        # no object detected
                        masks = None
                        prev_num = 0
                    else:
                        masks, _, _, object_tokens ,upscaled_embedding= predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False,
                        )
                        object_num[bs]=upscaled_embedding.shape[0]
                        upscaled_embedding_all[bs,0:int(object_num[bs]),:,:,:]=upscaled_embedding
                        #print("object_tokens")
                        #print(object_tokens.shape)

                        masks=masks.float()
                        mask_all[bs,:,:]=masks.sum(dim=0)
                        observation_tokens[bs,0:object_tokens.shape[0],:]=object_tokens

                        if frame_id == 0:
                            # memory initialization
                            #if bs==0:
                                #print("initialization\n\n")
                                #mem.initialization()
                                
                                #memory_shape = (batch_size, args.memory_len, mem.object_num, args.embed_dim)
                                #memory_table_shape = (batch_size,mem.object_num)
                                #mem.memory = torch.zeros(memory_shape)
                                #mem.memory_table = torch.zeros(memory_table_shape)
                            for i in range(mem.object_num):
                                color_dict[str(i)]=color_list[i]
                            for j in range(object_tokens.shape[0]):
                                mem.memory[bs,0,j,:]=object_tokens[j,:]
                                mem.memory_table[bs,j]=1
                            prev_num = mem.object_num
                            num_occupied=object_tokens.shape[0]
                        #else:
                            #memory, num_occupied, color_dict = match_from_mem(prev_num, object_tokens,mem.memory[bs],mem.memory_table[bs],frame_id, num_occupied, #color_list)
                            #mem.memory[bs,frame_id-1,:,:]=memory[frame_id-1,:,:]
                            #prev_num = object_tokens.shape[0]
                    # draw output image
                    if bs ==vis_batch:
                        masks_show=masks
                        boxes_filt_show=boxes_filt
                        pred_phrases_show= pred_phrases
                        image_show=image_input

                    video_id+=1
                observation_tokens=observation_tokens.to(device)
                pred=observation_tokens
                if frame_id>0:
                    pred,attn_index=mem(observation_tokens)
                    pred=pred.to(device)
                    attn_index=attn_index.to(device)
                else:
                    mem.initialization_sam(observation_tokens,observation_tokens.shape[1])

                #visualization
                indices=[]
                if frame_id>0:
                    indices=torch.argmin(attn_index.transpose(0, 1), dim=-1)
                    indices=indices[:,vis_batch]
                img=visualization_gif(mem,indices,image_show,masks_show,boxes_filt_show, pred_phrases_show,frame_id,output_dir)
                img = Image.fromarray(img, mode='RGB')
                gif.append(img)

                if frame_id>=3:
                    pred_mask=mask_decoder(sam.mask_decoder.output_hypernetworks_mlps[0],pred,upscaled_embedding_all,object_num,predictor).to(device)
                    smooth=0.001
                    mask_all=mask_all.float()#mask_all 0,1 type
            

                    #loss compute
                    loss_value_EM=EM_loss(pred,observation_tokens,attn_index)
                    loss_list_EM[frame_id-3]=loss_value_EM
                            
                    if frame_id>video_lenth:
                        break

            
            loss=loss_list_EM.mean()
            print("loss")
            print(loss)
            loss_list_all[epoch]=loss
            print(loss_list_all[0:(epoch+1)])
            optimizer.zero_grad()
            #for name, parms in mem.named_parameters():	
                #if name == "roll_out_module.transformer.h":
                    #print('-->name:', name)
                #    print('-->para:', parms)
            #        print('-->grad_requirs:',parms.requires_grad)
                    #print('-->grad_value:',parms.grad)
            #        print("===")
            loss.backward()
            optimizer.step()

            global_step=epoch + batch
            writer.add_scalar('TRAIN/loss', loss.item(), global_step)
            

            #save the gif
            os.makedirs("gif", exist_ok=True)
            frame_one = gif[0]
            frame_one.save(f"./gif/video_{epoch}.gif", format="GIF", append_images=gif,
               save_all=True, duration=100, loop=0)


        checkpoint = {
            'model': mem.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(args.log_path, 'checkpoint.pt.tar'))

        
    writer.close()
  
