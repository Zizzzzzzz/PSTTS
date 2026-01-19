# PSTTS: A Plug-and-Play Token Selector for Efficient Event-based Spatio-temporal Representation Modeling

## Environment for Testing:

- Python 3.8

  - `conda create -n your_env_name python=3.8`

- torch 1.13.1 + cu116
  - `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 `

- Install `mmcv-full`
  
  - `pip install openmim chardet`
  - `mim install mmcv==2.0.0rc4`
  
- Requirements

  - `cd DailyDVS-200-PSTTS/models/mmaction2_lastest`
  - `pip install -r requirements.txt`

- Install `mmaction2`

  - `pip install -e .`

## Prepare Data:

### DailyDVS-200:
- Download the DailyDVS-200(https://github.com/QiWang233/DailyDVS-200) dataset.

- Generate event frame sequences

  - `cd DailyDVS-200-PSTTS/utils`
  - `python3 preprocess`
    
- Generate temporal continuity maps

  - `python3 preprocess_TCM`


### HARDVS:
- Download the HARDVS(https://github.com/Event-AHU/HARDVS) dataset.

- Generate event frame sequences

  - `cd HARDVS-PSTTS/utils`
  - `python3 preprocess_HARDVS`
    
- Generate temporal continuity maps

  - `python3 preprocess_HARDVS_TCM`


## Test:

### DailyDVS-200:

- Download the UniformerV2(https://drive.google.com/file/d/15bDFzdiPUn28hyTU1Xyd9wG2EG9WySca/view?usp=drive_link) model weight and place it in the DailyDVS-200-PSTTS/pth. 
- `cd DailyDVS-200-PSTTS/models/mmaction2_lastest`
- `python3 tools/test.py ./configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_3xb8-u8_dailydvs200-rgb.py ../../pth/best_uniformerv2_dailydvs200.pth`

- Download the VideoSwin(https://drive.google.com/file/d/18iqUhrfmKpeV8GQluCeqqVq_-R8rQIDL/view?usp=drive_link) model weight and place it in the DailyDVS-200-PSTTS/pth. 
- `cd DailyDVS-200-PSTTS/models/mmaction2_lastest`
- `python3 tools/test.py ./configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_dailydvs200-rgb.py ../../pth/best_videoswin_dailydvs200.pth`

### HARDVS:

- Download the UniformerV2(https://drive.google.com/file/d/1UWSZ5dxRVhuGGUNs4eDG0-ODnOK6E_b4/view?usp=drive_link) model weight and place it in the HARDVS-PSTTS/pth. 
- `cd HARDVS-PSTTS/models/mmaction2_lastest`
- `python3 tools/test.py ./configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_3xb8-u8_hardvs.py ../../pth/best_uniformerv2_hardvs.pth`

- Download the VideoSwin(https://drive.google.com/file/d/1aDKkaAFWsT6cfVPGgaSy0FTCvWf749ZV/view?usp=drive_link) model weight and place it in the HARDVS-PSTTS/pth. 
- `cd HARDVS-PSTTS/models/mmaction2_lastest`
- `python3 tools/test.py ./configs/recognition/swin/swin-base-p244-w877_in1k-pre_1xb16-amp-32x2x1-30e_hardvs-rgb.py ../../pth/best_videoswin_hardvs.pth`