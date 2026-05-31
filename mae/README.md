# mae_pretrain_aug_0.259_loss.pth
- D=192, 3 block encoder
- bug with positional embeddings BEFORE projection into D
- no data augmentation

# mae_pretrain_aug_0.208_192
- same as above but with data agumentation (random cropping and flip)

# next
- Increased D=192->384, blocks 3->6 encoder
- adding positional embeddings AFTER projecting

# 3 ViT Encoder Blocks, 1 Decoder, D=192, Batch=256
- linear probe of 

# 6 ViT Encoder Blocks, 1 Decoder, D=384, Batch=256
- SSL training loss of 0.19122
- linear probe accuracy of 0.6326

NEXT:
- Fine-tune the linear probe on 8e_2d & visualize & Record!
- try torch.autocast to float16, then try full 12 layer ViT-Small! 
---> See how well that performs
- train ViT-Small fully supervised on STL10 (no unsupervised dataset)
- train a ResNet Tiny fully supervised -- keep practicing! Each time from scratch, keep practicing!
- Create a short report on my ablation studies
- Create a list of tricks section of the report for myself
- Dive into the hyperparameters: AdamW, linear probe training too
