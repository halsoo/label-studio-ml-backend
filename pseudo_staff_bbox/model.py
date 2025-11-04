import os
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import cv2


class PseudoStaffBBoxModel(LabelStudioMLBase):

  def setup(self):
    self.set("model_version", "0.0.1")
    self.staff_order = os.environ.get('STAFF_ORDER', "violin,violin,viola,cello").split(',')

  def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
    """
    Write your inference logic here
    :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
    :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
    :return model_response
      ModelResponse(predictions=predictions) with
      predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
    """

    predictions = []
    for task in tasks:
      sample = {
        "model_version": self.get("model_version"),
        "result":[]
      }

      path = self.get_local_path(task['data']['image'], task_id=task['id'])
      img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
      h, w = img.shape[:2]

      staff_order = self.staff_order
      n_staffs = len(staff_order)

      for i, staff_name in enumerate(staff_order):
        y1 = int(i * h / n_staffs)
        y2 = int((i + 1) * h / n_staffs)

        bbox = {
          "from_name": 'label',
          "to_name": 'image',
          "type": "rectanglelabels",
          "value": {
            "original_width": w,
            "original_height": h,
            "x": 0,
            "y": y1/h * 100,
            "width": 100,
            "height": (y2-y1)/h * 100,
            "rectanglelabels": [staff_name],
          }
        }

        sample["result"].append(bbox)

      predictions.append(sample)

    return ModelResponse(predictions=predictions)


  def fit(self, event, data, **kwargs):
    print('This model does not need training, skipping fit step.')