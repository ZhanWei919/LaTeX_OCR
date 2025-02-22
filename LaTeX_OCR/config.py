# 数据路径
data_name = 'CROHME'
vocab_path = 'data/full/vocab.json'
train_set_path = 'data/full/train.json'
val_set_path = 'data/full/val.json'

# 模型参数
emb_dim = 128       
attention_dim = 256 
decoder_dim = 512   
dropout = 0.3       
buckets = [[160, 80], [240, 100], [320, 120], [400, 120], [480, 120], [560, 120]]  

# 训练参数
start_epoch = 0
epochs = 256
epochs_since_improvement = 0
batch_size = 128  
accumulation_steps = 4  
num_workers = 11    
encoder_lr = 1e-4
decoder_lr = 1e-4
lr_scheduler = 'ReduceLROnPlateau'  
lr_patience = 10
lr_factor = 0.5
grad_clip = 6
label_smoothing = 0.1
alpha_c = 0.5       
best_score = 0.
print_freq = 1
checkpoint = None
save_freq = 2
pin_memory = True
prefetch_factor = 2
persistent_workers = True


use_amp = True  
use_compile = False  
compile_mode = "reduce-overhead"  

# 梯度检查点
use_checkpoint = True
checkpoint_layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']  

# AST 相关配置
ast_vocab_size = 1000
ast_embed_dim = 128   
ast_hidden_dim = 256  
fusion_dim = 128     