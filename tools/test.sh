CUDA_VISIBLE_DEVICES=8 python test.py --file_root SYSU --savedir /home/gaoyuhao/paper/denseNet/tools/123 
CUDA_VISIBLE_DEVICES=8 python test.py --file_root BCDD --savedir /home/gaoyuhao/paper/denseNet/tools/123 
# CUDA_VISIBLE_DEVICES=8 python test.py --file_root SYSU --savedir /home/gaoyuhao/paper/denseNet/tools/results_LEVIR21010model_ccf_mul_msff_m_unet_cca_esa_regnet1.6_nomod__d8_iter_45000_lr_0.0005 --lr 5e-4 --max_steps 40000
# python test.py --file_root BCDD --lr 5e-4 --max_steps 40000
# python test.py --file_root SYSU --lr 5e-4 --max_steps 40000