import logging
import typing

import numpy as np
import pandas as pd


class CandleSticks:

    def __init__(self, days: pd.DataFrame, points: np.ndarray):
        """
        :param days: A single field data frame.  The field must be a field of UNIX times, and
                     must be named 'epochmilli'.  Candle stick values will be evaluated at
                     the 'epochmilli' field's UNIX time points.

        :param points: The required candle stick tile values

        logging:
            logging.basicConfig(level=logging.INFO)
            logging.disable(logging.WARN)
        """
        logging.disable(logging.WARN)
        self.logger = logging.getLogger(__name__)

        self.days = days
        self.points = points

    def quantiles(self, data: pd.DataFrame, fields: typing.List):
        """
        :param data: The DataFrame that hosts the data that will be used for quantile calculations
        :param fields: The DataFrame fields that will be used for the quantile calculations
        """

        values = data[fields].quantile(q=self.points, axis=0)
        values = values.transpose()

        return values

    @staticmethod
    def tallies(data: pd.DataFrame, fields: typing.List) -> pd.Series:
        """
        :param data: The DataFrame that hosts the data that will be used for sum calculations
        :param fields: The DataFrame fields that will be used for the sum calculations
        """
        values = data[fields].sum().rename('tally')

        return values

    def nonzero(self, data: pd.DataFrame, fields: typing.List):
        """
        :param data: The DataFrame that hosts the data that will be used for nonzero determinations
        :param fields: The DataFrame fields that will be used for nonzero calculations
        """
        values = (data[fields] != 0).sum().rename('nonzero')
        self.logger.info(values)
        return values

    def sticks(self, instances):
        """

        :param instances:
        :return:
        """

        values = self.days.merge(instances, how='inner', left_on='epochmilli', right_index=True)

        return values

    def execute(self, data, fields):
        """

        :param data: The DataFrame that hosts the data that will be used for the calculations herein
        :param fields: The fields of values
        :return:
        """

        quantiles = self.quantiles(data, fields)
        tallies = self.tallies(data, fields)
        tallies = tallies.astype(dtype=np.int64)
        nonzero = self.nonzero(data, fields)
        maxima = data[fields].max(axis=0).rename('max')
        instances = pd.concat([quantiles, maxima, tallies, nonzero], axis=1)

        instances.reset_index(drop=False, inplace=True)

        return instances
