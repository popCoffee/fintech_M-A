from django.db.models import Model

from risk_reporting.deal_downside.strategies.downside_strategy import DownsideStrategy


class DownsideCalculationContext:
    def __init__(self, strategy: DownsideStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    def set_strategy(self, strategy: DownsideStrategy):
        self._strategy = strategy

    def get_new_data(self, params=None):
        return self._strategy.get_new_data(params)

    def prepare_data(self, data):
        self._strategy.prepare_data(data)

    def generate_model(self, params=None):
        generated = self._strategy.generate_model(params)
        if not generated:
            return None
        return self._strategy

    def calculate_downside(self, data, params=None):
        self._strategy.prepare_data(data)
        generated = self._strategy.generate_model(params)
        if not generated:
            print("Model could not be generated")
            return None
        return self._strategy.calculate_downside(self.strategy.get_new_data())

    def calculate_downside_from_model(self, data=None) -> float:
        if data is not None:
            return self._strategy.calculate_downside(data)
        else:
            return self._strategy.calculate_downside(self._strategy.get_new_data())

    def save_model(self) -> Model:
        """ function to save exisiting parameters needed to calculate downside to database
            returns the model instance saved to the database
        """
        return self._strategy.save_model()

    def load_model(self, db_model_data):
        self._strategy.load_model(db_model_data)
