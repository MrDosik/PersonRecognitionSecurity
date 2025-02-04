# you are encouraged to read `notes_on_mr.md` first :)
import torch
import torch.backends.cudnn as cudnn

import numpy as np 
import os 
import argparse
import time
from tqdm import tqdm
import joblib

from utils.setup import get_model,get_data_loader
from utils.defense import gen_mask_set,double_masking_precomputed,certify_precomputed

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help="directory of checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="directory of data")
parser.add_argument('--dataset', default='imagenette',type=str,choices=('imagenette','imagenet','cifar','cifar100','svhn','flower102'),help="dataset")
parser.add_argument("--model",default='vit_base_patch16_224',type=str,help="model name")
parser.add_argument("--num_img",default=-1,type=int,help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride",default=-1,type=int,help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask",default=-1,type=int,help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size",default=32,type=int,help="size of the adversarial patch (square patch)")
parser.add_argument("--pa",default=-1,type=int,help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb",default=-1,type=int,help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir",default='dump',type=str,help='directory to dump two-mask predictions')
parser.add_argument("--override",action='store_true',help='override dumped file')

args = parser.parse_args()
DATASET = args.dataset
MODEL_DIR = os.path.join('.',args.model_dir)
DATA_DIR = os.path.join(args.data_dir,DATASET)
DUMP_DIR = os.path.join('.',args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

MODEL_NAME = args.model
NUM_IMG = args.num_img

#get model and data loader
model = get_model(MODEL_NAME,DATASET,MODEL_DIR)
val_loader,NUM_IMG,ds_config = get_data_loader(DATASET,DATA_DIR,model,batch_size=16,num_img=NUM_IMG,train=False)

device = 'cuda' 
model = model.to(device)
model.eval()
cudnn.benchmark = True
num_classes = 1000 if args.dataset == 'imagenet' else 10

# generate the mask set
mask_list,MASK_SIZE,MASK_STRIDE = gen_mask_set(args,ds_config)
#print(len(mask_list))
#args.num_mask = int((len(mask_list))**0.5)
# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
SUFFIX = '_one_mask_{}_{}_m{}_s{}_{}.z'.format(DATASET,MODEL_NAME,MASK_SIZE,MASK_STRIDE,NUM_IMG)
SUFFIX2 = '_two_mask_{}_{}_m{}_s{}_{}.z'.format(DATASET,MODEL_NAME,MASK_SIZE,MASK_STRIDE,NUM_IMG)
print(os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX2))
if not args.override and os.path.exists(os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX)):
    print('loading one-mask predictions')
    confidence_map_list = joblib.load(os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX))
    prediction_map_list = joblib.load(os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX))
    orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    label_list = joblib.load(os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
elif not args.override and os.path.exists(os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX2)):
    print('loading two-mask predictions')
    confidence_map_list = joblib.load(os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX2))
    prediction_map_list = joblib.load(os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX2))
    print('converting two-mask predictions to one-mask predictions')
    confidence_map_list = np.diagonal(confidence_map_list,axis1=1,axis2=2)
    prediction_map_list = np.diagonal(prediction_map_list,axis1=1,axis2=2)
    orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    label_list = joblib.load(os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
else:
    print('computing one-mask predictions')
    prediction_map_list = []
    confidence_map_list = []
    label_list = []
    orig_prediction_list = []
    for data,labels in tqdm(val_loader):
        data=data.to(device)
        labels = labels.numpy()
        num_img = data.shape[0]

        #two-mask predictions
        num_mask = len(mask_list)
        prediction_map = np.zeros([num_img,num_mask],dtype=int)
        confidence_map = np.zeros([num_img,num_mask])

        for i,mask in enumerate(mask_list):

            masked_output = model(torch.where(mask,data,torch.tensor(0.)))
            masked_output = torch.nn.functional.softmax(masked_output,dim=1)
            masked_conf, masked_pred = masked_output.max(1)
            masked_pred = masked_pred.detach().cpu().numpy()
            masked_conf = masked_conf.detach().cpu().numpy()

            prediction_map[:,i] = masked_pred
            confidence_map[:,i] = masked_conf  
                    
        #vanilla predictions
        clean_output = model(data)
        clean_conf, clean_pred = clean_output.max(1)  
        clean_pred = clean_pred.detach().cpu().numpy()
        orig_prediction_list.append(clean_pred)
        prediction_map_list.append(prediction_map)
        confidence_map_list.append(confidence_map)
        label_list.append(labels)
    
    prediction_map_list = np.concatenate(prediction_map_list)
    confidence_map_list = np.concatenate(confidence_map_list)
    orig_prediction_list = np.concatenate(orig_prediction_list)
    label_list = np.concatenate(label_list)

    joblib.dump(confidence_map_list,os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX))
    joblib.dump(prediction_map_list,os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX))
    joblib.dump(orig_prediction_list,os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    joblib.dump(label_list,os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))


def provable_detection(prediction_map,confidence_map,label,orig_pred,tau):
    if orig_pred != label: # clean prediction is incorrect
        return 0,0 # 0 for incorrect clean prediction
    clean = 1
    provable = 1
    if np.any(np.logical_or(prediction_map!=label,confidence_map<tau)):
        provable = 0
    if np.any(np.logical_and(prediction_map!=label,confidence_map>tau)):
        clean = 0
    return provable,clean


clean_list = []
robust_list = []
for tau in np.arange(0.,1.,0.05):
    print(tau)
    clean_corr = 0
    robust_cnt = 0 
    vanilla_corr =0 
    for i,(prediction_map,confidence_map,label,orig_pred) in enumerate(zip(prediction_map_list,confidence_map_list,label_list,orig_prediction_list)):
        #print(confidence_map)
        provable,clean = provable_detection(prediction_map,confidence_map,label,orig_pred,tau)
        robust_cnt+=provable
        clean_corr+=clean
        vanilla_corr +=orig_pred==label
    robust_list.append(robust_cnt/NUM_IMG)
    clean_list.append(clean_corr/NUM_IMG)
    print("Provable robust accuracy ({}):".format(tau),robust_cnt/NUM_IMG)
    print("Clean accuracy with defense:",clean_corr/NUM_IMG)
    print("Clean accuracy without defense:",vanilla_corr/NUM_IMG)
    print()
print('clean_list=', [100*x for x in clean_list])
print('robust_list=',[100*x for x in robust_list])