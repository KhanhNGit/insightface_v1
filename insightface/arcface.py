from __future__ import division
import cv2
import torch
import numpy as np

# from .face_align import norm_crop

class ArcFaceONNX:
    def __init__(self, model_file=None):
        assert model_file is not None
        self.model_file = model_file
        self.taskname = 'recognition'
        # find_sub = False
        # find_mul = False
        # for nid, node in enumerate(graph.node[:8]):
        #     #print(nid, node.name)
        #     if node.name.startswith('Sub') or node.name.startswith('_minus'):
        #         find_sub = True
        #     if node.name.startswith('Mul') or node.name.startswith('_mul'):
        #         find_mul = True
        # if find_sub and find_mul:
        #     #mxnet arcface model
        #     input_mean = 0.0
        #     input_std = 1.0
        # else:
        #     input_mean = 127.5
        #     input_std = 127.5
        self.input_mean = 0.0
        self.input_std = 1.0
        #print('input mean and std:', self.input_mean, self.input_std)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(self.model_file).to(self.device)
        self.model.eval()
        self.input_size = (112, 112)

    def get(self, face):
        # aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        # face.embedding = np.array(self.get_feat(aimg)).flatten()
        # return face.embedding
        return np.array(self.get_feat(face))

    # def compute_sim(self, feat1, feat2):
    #     from numpy.linalg import norm
    #     feat1 = feat1.ravel()
    #     feat2 = feat2.ravel()
    #     sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    #     return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        blob = torch.from_numpy(blob)
        blob = blob.to(self.device)
        with torch.no_grad():
            net_out = self.model(blob)
            net_out = [i.detach().cpu().numpy() for i in net_out]
        return net_out

    # def forward(self, batch_data):
    #     blob = (batch_data - self.input_mean) / self.input_std
    #     net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
    #     return net_out