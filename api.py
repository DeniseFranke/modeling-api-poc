from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
from sklearn.externals import joblib as jl



#Present it in Flask API
app = Flask(__name__)
model_file_name = "titanic_model.pkl"
model_columns_file_name = "titanic_model_columns.pkl"

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # default port value if none provided
    lr = jl.load(model_file_name) # Load "titanic_model.pkl"
    print (lr)
    print ('Model loaded')
    model_columns = jl.load(model_columns_file_name) # Load "titanic_model_columns.pkl"
    print (model_columns)
    print ('Model columns loaded')
    app.run(port=port, debug=True)

