"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

  ```bash
  pip install -r requirements-test.txt
  ```
Then execute `pytest` in the directory of this file.

- Change `PseudoStaffBBoxModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import pytest
import json
from model import PseudoStaffBBoxModel


@pytest.fixture
def client():
  from _wsgi import init_app
  app = init_app(model_class=PseudoStaffBBoxModel)
  app.config['TESTING'] = True
  with app.test_client() as client:
    yield client


def test_predict(client):
  request = {
    'tasks': [{
      'id': 'some-unique-id',
      'data': {
        "image": "https://drive.google.com/file/d/17GgCIcxwOVsVRT64OE4wwjZUVE827UeQ"
      }
    }],
    # Your labeling configuration here
    'label_config': '''
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="violin" background="#FFA39E"/>
        <Label value="viola" background="#49d30d"/>
        <Label value="cello" background="#6b9cff"/>
      </RectangleLabels>
    </View>
    '''
  }

  expected_response = {
    'results': [{
    }]
  }

  response = client.post('/predict', data=json.dumps(request), content_type='application/json')
  assert response.status_code == 200
  response = json.loads(response.data)
  assert response == expected_response
