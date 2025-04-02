import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request
import pandas as pd
from predict_inventory import predict_inventory  # Import your prediction function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    explanation = (
        "This model predicts the future stock (inventory level) for a product based on historical data "
        "such as sales, current stock levels, and delivery times. It uses a Linear Regression model that was "
        "trained on historical data. Enter a date below (in YYYY-MM-DD format) to see the predicted inventory level for that day."
    )
    sample_date = None

    if request.method == 'POST':
        # Get the date entered by the user
        user_date_str = request.form.get('date')
        try:
            user_date = datetime.strptime(user_date_str, "%Y-%m-%d")
            sample_date = user_date_str
        except ValueError:
            sample_date = "Invalid date provided"
            user_date = None

        # Load a sample row from your cleaned data as the base input for prediction
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/cleaned_data.csv')
        df = pd.read_csv(data_path)

        # Use the first row as a sample input
        sample_row = df.iloc[[0]].copy()

        # Update the 'date' field with the user-specified date.
        # For demonstration, also update the 'delivery_time' to be 10 days later.
        if user_date:
            sample_row['date'] = user_date_str
            sample_row['delivery_time'] = (user_date + timedelta(days=10)).strftime("%Y-%m-%d")
        else:
            sample_date = "Invalid date provided"

        # Get the prediction from your model
        prediction = predict_inventory(sample_row)[0]

    return render_template('index.html', prediction=prediction, explanation=explanation, sample_date=sample_date)

if __name__ == '__main__':
    app.run(debug=True)
