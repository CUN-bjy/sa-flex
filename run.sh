CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n5 --num-slots 5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n10 --num-slots 10 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n15 --num-slots 15 --seed 0

# CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n5 --num-slots 5 --seed 1
# CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n10 --num-slots 10 --seed 1
# CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n15 --num-slots 15 --seed 1

# CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n5 --num-slots 5 --seed 2
# CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n10 --num-slots 10 --seed 2
# CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --prefix sa-clevr-n15 --num-slots 15 --seed 