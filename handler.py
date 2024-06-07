import json
import os
import pickle

import pandas as pd
import torch


class TitanicHandler(object):
    def __init__(self):
        super(TitanicHandler, self).__init__()
        self.model = None
        self.preprocessor = None

    def initialize(self, context):
        """Initialize the model and preprocessor"""
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load the model
        model_path = os.path.join(model_dir, "titanic_model.pt")
        self.model = torch.jit.load(model_path)

        # Load the preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        with open(preprocessor_path, "rb") as reader:
            self.preprocessor = pickle.load(reader)

        self.model.eval()

    def preprocess(self, data):
        """Preprocess the input data"""
        # Extract data from the request
        inputs = data[0].get("data")
        if inputs is None:
            inputs = data[0].get("body")
        if isinstance(inputs, (bytes, bytearray)):
            inputs = inputs.decode("utf-8")
        # Convert input JSON to pandas DataFrame
        inputs = pd.read_json(json.dumps(inputs), orient="records")
        # Apply preprocessing
        preprocessed_data = self.preprocessor.transform(inputs)

        # Convert to torch tensor
        preprocessed_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)

        return preprocessed_tensor

    def inference(self, data):
        """Run inference on the preprocessed data"""
        with torch.no_grad():
            predictions = self.model(data)
            # Convert probabilities to binary predictions
            predictions = (predictions > 0.5).int()
        return predictions

    def postprocess(self, inference_output):
        """Postprocess the output to return a suitable format"""
        # Convert predictions to a list of dictionaries
        result = inference_output.numpy().tolist()
        result = [[{"survived": int(pred[0])} for pred in result]]
        return result

    def handle(self, data, context):
        """Entry point for TorchServe custom handler"""
        # Preprocess
        preprocessed_data = self.preprocess(data)

        # Inference
        predictions = self.inference(preprocessed_data)

        # Postprocess
        response = self.postprocess(predictions)

        return response
