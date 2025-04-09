from flask import Blueprint, request, jsonify
from src.backend.controllers.prediction_controller import predict_inventory

prediction_bp = Blueprint('prediction_bp', __name__)

@prediction_bp.route('/predict', methods=['GET'])
def predict():
    try:
        day = int(request.args.get('day', 0))
        prediction = predict_inventory(day)
        return jsonify({'Прогнозируемый запас': prediction})
    except ValueError:
        return jsonify({'error': 'Неверный формат дня. Должно быть положительное число.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
