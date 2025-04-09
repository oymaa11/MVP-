from flask import Flask
from src.backend.routes.prediction_routes import prediction_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(prediction_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
