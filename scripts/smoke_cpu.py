"""Offline synthetic MAT -> original node grid -> unit fingerprint template.

This checks software integration only. It does not reproduce recognition metrics.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLBACKEND','Agg')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,tempfile
import numpy as np
import torch
from scipy.io import savemat
from MyModel.datasets import FingerprintDataset
from MyModel.model import FingerprintNet
from MyModel.utils.utils import collate_list,move_sample_to_device


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=None);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix='d2s_smoke_') as temp:
        folder=Path(temp)/'cfpose';folder.mkdir();path=folder/'121_1.mat'
        y,x=np.mgrid[:640,:480];image=(127+70*np.sin(x/6.+y/30.)).astype(np.float32)
        mask=(((x-240)/185.)**2+((y-320)/280.)**2<1).astype(np.float32)
        savemat(path,{'img':image,'mask':mask})
        dataset=FingerprintDataset(files=[str(path)],data_name='cfpose',mode='test',augment=False,use_minu_metadata=False)
        batch=collate_list([dataset[0]])
        assert batch['coords'].shape==(1,1131,2),batch['coords'].shape
        model=FingerprintNet(feat_dim=128,num_layers=3,knn_k=25,topk_pool=50,tensor_neurons=32,pretrained_backbone=False).eval()
        with torch.no_grad():template,_,debug=model(move_sample_to_device(batch,torch.device('cpu')),return_debug=True)
        assert template.shape==(1,128) and torch.isfinite(template).all()
        assert abs(float(template.norm())-1)<1e-5
        assert len(debug['selected_idx'][0])==50
        result=dict(status='PASS',device='cpu',network_downloads=False,synthetic_data=True,
                    input_shape=list(batch['full_img'].shape),nodes=1131,selected_nodes=50,
                    template_shape=list(template.shape),template_norm=float(template.norm()),
                    training_updates=0,recognition_metrics_reproduced=False,
                    torch=torch.__version__,numpy=np.__version__)
        if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
