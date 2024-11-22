import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        percentage_to_use = 1 # Use 100% of the data

        with open('./dataset/prompt.json', 'rt') as f:
            all_data = [json.loads(line) for line in f]
            for item in all_data:
                item['target'] = item['target'].replace('.png', '.jpg')

            num_samples = int(len(all_data) * percentage_to_use)
            self.data = all_data[:num_samples]

        #print(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = ""#item['prompt']
        source = cv2.imread('./dataset/' + source_filename)
        target = cv2.imread('./dataset/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Resize the images to 512x512
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

dataset = MyDataset()
print(len(dataset))
# item = dataset[100]
# jpg = item['jpg']
# txt = item['txt']
# hint = item['hint']
# cv2.imwrite("output_depth_1.png",jpg)
# cv2.imwrite("input_depth_1.png",hint)
# print(txt)
# print(jpg.shape)
# print(hint.shape)
# exit()
# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


if __name__ == '__main__':
    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)

