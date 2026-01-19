import os
root = '../labels'
splits = os.listdir(root)
for split in splits:
    # 定义文件路径
    category_file = os.path.join(root, split)  # 存储“序列名称 图像数量”的文件
    image_count_file = 'framenum.txt'       # 存储“序列名称 类别”的文件
    output_file = os.path.join('../full_labels', split)# 'merged_info.txt'      # 合并后的输出文件

    # 读取 image_count.txt 文件，存储为字典
    image_count_dict = {}
    with open(image_count_file, 'r') as f:
        for line in f:
            seq_name, count = line.strip().split()  # 分割序列名称和图像数量
            image_count_dict[seq_name] = count

    # 读取 category.txt 文件，合并信息并写入新文件
    with open(category_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            seq_name, category = line.strip().split()  # 分割序列名称和类别
            # 从 image_count_dict 中获取对应的图像数量
            if seq_name in image_count_dict:
                count = image_count_dict[seq_name]
                # 将合并后的信息写入新文件
                out_f.write(f"{seq_name} {count} {category}\n")
            else:
                print(f"Warning: Sequence '{seq_name}' not found in image_count.txt")

    print(f"合并完成，结果已保存到 {output_file}")