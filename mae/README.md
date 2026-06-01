# 3 ViT, 1 Decoder no data augmentation
- SSL 0.259
- bug with positional embeddings BEFORE projection into D

# 3 ViT, 1 Decoder mae_pretrain_aug_0.208_192, 200 epochs
- SSL 0.208 training loss
- same as above but with data agumentation (random cropping and flip)

# 6 ViT Encoder Blocks, 1 Decoder, D=384, Batch=256, 200 epochs
- SSL training loss of 0.19122
- linear probe accuracy of 0.6326

# 8 ViT Encoder Blocks, 2 Decoder, D=384, Batch=256, 200 epochs
- SSL training loss of 0.1848
- linear probe accuracy of 0.6787
- D=384, D=384 AND 3 heads

# ViT-Base, 8 Decoder, 400 epochs, Autocast, Gradient Checkpointing
- TODO!


NEXT:
- X Fine-tune the linear probe on 8e_2d & visualize & Record!
- try torch.autocast to float16, then try full 12 layer ViT-Small! 
---> See how well that performs
- train ViT-Small fully supervised on STL10 (no unsupervised dataset)
- train a ResNet Tiny fully supervised -- keep practicing! Each time from scratch, keep practicing!
- Create a short report on my ablation studies
- Create a list of tricks section of the report for myself
- Dive into the hyperparameters: AdamW, linear probe training too

- TRY gradient checkpointing!
- Try 6 heads