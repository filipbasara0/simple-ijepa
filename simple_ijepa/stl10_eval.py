import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from simple_ijepa.utils import inference_transforms

import warnings

warnings.filterwarnings("ignore")


def logistic_regression(embeddings, labels, embeddings_val, labels_val):
    X_train, X_test = embeddings, embeddings_val
    y_train, y_test = labels, labels_val
    
    clf = LogisticRegression(max_iter=100)
    clf = CalibratedClassifierCV(clf)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy STL10: ", acc)


class STL10Eval:

    def __init__(self, image_size=96):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = inference_transforms(img_size=(image_size, image_size))
        train_ds = torchvision.datasets.STL10("data/",
                                        split='train',
                                        transform=transform,
                                        download=True)
        val_ds = torchvision.datasets.STL10("data/",
                                        split='test',
                                        transform=transform,
                                        download=True)
    

        self.train_loader = DataLoader(train_ds,
                                batch_size=64,
                                num_workers=2)
        self.val_loader = DataLoader(val_ds,
                            batch_size=64,
                            num_workers=2)

    @torch.inference_mode
    def evaluate(self, scl_model):
        model = scl_model.target_encoder
        # model = scl_model.online_encoder
        embeddings, labels = self._get_image_embs_labels(model, self.train_loader)
        embeddings_val, labels_val = self._get_image_embs_labels(model, self.val_loader)
        logistic_regression(embeddings, labels, embeddings_val, labels_val)

    @torch.inference_mode
    def _get_image_embs_labels(self, model, dataloader):
        embs, labels = [], []
        for _, (images, targets) in enumerate(dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                out = model(images)
                features = out.cpu().detach()
                features = features.mean(dim = 1)
                embs.extend(features.tolist())
                labels.extend(targets.cpu().detach().tolist())
        return np.array(embs), np.array(labels)
