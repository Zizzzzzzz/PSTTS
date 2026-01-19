import os
from TCM_convert_snn import EventImageConvert

eventImageConvert = EventImageConvert(interval=0.5)
root = '/root/data1/dataset/DailyDvs-200' # Replace with your dataset path
classes = os.listdir(root)
for cla in classes:
    seqs = os.listdir(os.path.join(root, cla))
    for seq in seqs:
        event_file = os.path.join(root, cla, seq)
        
        output_root = os.path.join('/root/data1/dataset/DailyDvs-200-TCMframesnn', seq.split('.aedat4')[0])
        os.makedirs(output_root, exist_ok=True)
        aps_frames_NUM = eventImageConvert._get_frames_NUM(event_file)
        print(event_file, '--------', aps_frames_NUM)
        eventImageConvert._events_to_event_images(event_file, output_root, aps_frames_NUM=aps_frames_NUM, interval=0.5)