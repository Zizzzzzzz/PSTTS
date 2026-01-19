import os
from event_image_convert_HARDVS_TCM_snn import EventImageConvert

eventImageConvert = EventImageConvert(interval=0.25)
root = '/root/data1/dataset/MINIHARDVS_EVENT_files'
classes = os.listdir(root)
for cla in classes:
    seqs = os.listdir(os.path.join(root, cla))
    for seq in seqs:
        event_file = os.path.join(root, cla, seq)
        
        output_root = event_file.replace('MINIHARDVS_EVENT_files', 'MINIHARDVS_EVENT_files-frames0p25-TCMsnn')
        os.makedirs(output_root, exist_ok=True)

        event_file_name = os.path.join(root, cla, seq, seq+'.npz')
        aps_frames_NUM = eventImageConvert._get_frames_NUM(event_file_name)
        print(event_file, '--------', aps_frames_NUM)
        eventImageConvert._events_to_event_images(event_file_name, output_root, aps_frames_NUM=aps_frames_NUM, interval=0.25)