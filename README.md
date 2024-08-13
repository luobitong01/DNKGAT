# DNKGAT: Dynamic Neighbor-enhanced Knowledge Graph Attention Network for Rumor Detection
pheme_clean, pheme_concept_yago, pheme_entity and pheme_temporal_data these four folders are packed into a zip file, and can be obtained from https://www.dropbox.com/s/xwn5dvqgx2n2vsd/pheme_peocessed_data.zip?dl=0. The Raw Pheme dataset can be obtained from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 (or https://www.dropbox.com/s/j8x105s60ow997f/all-rnr-annotated-threads.zip?dl=0).
## Run
```
conda create -n dnkgt python=3.10.11
activate ddgcn
pip install -r requirement.txt -i https://mirrors.aliyun.com/pypi/simple
cd whls
pip install torch_cluster-1.6.0+pt113cu117-cp310-cp310-....whl
pip install torch_scatter-2.1.0+pt113cu117-cp310-cp310-....whl
pip install torch_sparse-0.6.16+pt113cu117-cp310-cp310-....whl
pip install torch_spline_conv-1.2.1+pt113cu117-cp310-cp310-....whl
pip install torch-geometric==2.2.0 -i https://mirrors.aliyun.com/pypi/simple
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 --default-timeout=1000

python train_dynamic.py --dataset pheme --cuda 0 --batch 32 --epoch 5 --lr 0.001 --dataset_dir your_dataset_dir
```
