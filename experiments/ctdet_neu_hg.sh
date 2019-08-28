cd src
# train
python main.py ctdet --exp_id neu_hg --arch resdcn_101 --batch_size 1 --lr 2.5e-4 --dataset neu_det --load_model ../models/ctdet_coco_resdcn101.pth --gpus 0
# test
python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..


python main.py ctdet --exp_id neu_resdcn_18 --arch resdcn_18 --batch_size 1 --lr 2.5e-4 --dataset neu_det --load_model ../models/ctdet_coco_resdcn18.pth --gpus 0
python main.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --num_epochs 70 --lr_step 45,60