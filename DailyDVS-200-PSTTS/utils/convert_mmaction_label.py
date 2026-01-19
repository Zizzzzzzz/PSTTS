import os

root = '../train_val_test'
splits = os.listdir(root)

save_root = '../labels'
for spl in splits:
    with open(os.path.join(root, spl), "r") as f:
        labels = f.readlines()
        with open(os.path.join(save_root, spl.split('.txt')[0]+'_label.txt'), "a") as fw:
            for label in labels:
                fw.write(label.split('/')[-1].replace('.aedat4', ''))
