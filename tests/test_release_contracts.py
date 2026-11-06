import ast
import unittest
from pathlib import Path
import argparse
import json
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


class ReleaseContractsTest(unittest.TestCase):
    def test_every_benchmark_consumer_accepts_rank1_rank5_rank10(self):
        bad=[]
        for path in [ROOT/'MyModel/scripts/train.py',ROOT/'MyModel/scripts/test.py']:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='test_benchmark':
                    target=node.targets[0]
                    if isinstance(target,(ast.Tuple,ast.List)) and len(target.elts)!=10:bad.append((path.name,node.lineno,len(target.elts)))
        self.assertEqual(bad,[])

    def test_empty_benchmark_has_the_same_ten_field_interface(self):
        from MyModel.scripts.test import test_benchmark
        result=test_benchmark(nn.Identity(),[],torch.device('cpu'),dataset_name='cfpose',enable_score_histogram=False)
        self.assertEqual(len(result),10)

    def test_explicit_offline_construction_keeps_the_paper_layer3_architecture(self):
        from MyModel.model.mymodel import FingerprintNet
        model=FingerprintNet(feat_dim=128,tensor_neurons=32,topk_pool=50,pretrained_backbone=False)
        self.assertEqual(tuple(model.encoder.cnn_proj.weight.shape),(128,256,1,1))
        self.assertEqual(tuple(model.encoder.backbone[0].weight.shape),(64,1,7,7))
        self.assertEqual(model.topk_pool,50)
        self.assertFalse(model.encoder.backbone[0].weight.requires_grad)
        self.assertTrue(model.encoder.backbone[1].weight.requires_grad)

    def test_command_line_overrides_explicit_config(self):
        from MyModel.utils.cli import parse_with_config
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'config.json';config.write_text(json.dumps({'epochs':3000}),encoding='utf-8')
            parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--epochs',type=int,default=200)
            with patch('sys.argv',['train','--config',str(config),'--epochs','1650']):
                self.assertEqual(parse_with_config(parser).epochs,1650)

    def test_unknown_config_keys_cannot_silently_change_a_recipe(self):
        from MyModel.utils.cli import parse_with_config
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'config.json';config.write_text('{"unknown_loss":1}',encoding='utf-8')
            parser=argparse.ArgumentParser();parser.add_argument('--config')
            with patch('sys.argv',['train','--config',str(config)]),self.assertRaises(SystemExit):parse_with_config(parser)

    def test_trusted_legacy_checkpoint_metadata_loads_on_current_torch(self):
        from MyModel.utils.utils import load_checkpoint_into_model
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trusted_test.pt';source=nn.Linear(2,3);target=nn.Linear(2,3)
            torch.save({'model':source.state_dict(),'legacy_numpy_metadata':np.array([42])},path)
            loaded=load_checkpoint_into_model(target,str(path),torch.device('cpu'))
            self.assertEqual(loaded['legacy_numpy_metadata'].tolist(),[42])
            torch.testing.assert_close(source.weight,target.weight)

    def test_gallery_changes_identification_not_verification_pairs(self):
        from MyModel.scripts.test import test_benchmark
        from scipy.io import savemat
        class FixedTemplates(nn.Module):
            def forward(self,batch,return_debug=False):
                features=torch.zeros(4,128);features[:,:2]=torch.tensor([[1.,0.],[.6,.8],[0.,1.],[.8,.6]])
                ix=batch['full_img'][:,0,0,0].long()
                return features[ix],None,{}
        batches=[]
        for i,(identity,view) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
            batches.append(dict(full_img=torch.full((1,1,2,2),float(i)),coords=torch.zeros(1,32,2),
                node_mask=torch.ones(1,32,dtype=torch.bool),W=torch.ones(1,32,32),
                id=torch.tensor([identity]),file_path=[f'/synthetic/cfpose/{identity}_{view}.mat']))
        options=dict(dataset_name='cfpose',use_ntn=False,enable_score_histogram=False)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'angles.mat';angle=np.full((2,10,3),50.,dtype=float)
            angle[0,0,1]=30.;angle[0,1,1]=1.;angle[1,0,1]=0.;angle[1,1,1]=20.
            savemat(path,{'angle_refine':angle})
            first=test_benchmark(FixedTemplates(),batches,torch.device('cpu'),gallery_strategy='first',**options)
            canonical=test_benchmark(FixedTemplates(),batches,torch.device('cpu'),gallery_strategy='angle_gt',gallery_metadata=str(path),**options)
        self.assertEqual(len(first),10);self.assertEqual(len(canonical),10)
        np.testing.assert_allclose(first[:7],canonical[:7],equal_nan=True)
        self.assertEqual(sum(first[2:6]),6)
        self.assertNotEqual(first[7],canonical[7])


if __name__=='__main__':unittest.main()
