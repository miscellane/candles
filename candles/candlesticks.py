import logging
import typing

import numpy as np
import pandas as pd


class CandleSticks:

    def __init__(self, days: pd.DataFrame, key: str, points: np.ndarray):
        """

        :param days: A frame summarising the days for which candle stick values will be evaluated.  The
                     frame must include an 'epochmilli' column, which encodes the dates in UNIX time form
        :param key: Join/Merge key for ...
        :param points: The required candle stick tile values
        """
        self.days = days
        self.key = key
        self.points = points

        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def quantiles(self, data: pd.DataFrame, fields: typing.List):
        """
        :param data: The DataFrame that hosts the data that will be used for quantile calculations
        :param fields: The DataFrame fields that will be used for the quantile calculations
        """

        values = data[fields].quantile(q=self.points, axis=0)
        values = values.transpose()
        self.logger.info(values)
        return values

    def tallies(self, data: pd.DataFrame, fields: typing.List):
        """
        :param data: The DataFrame that hosts the data that will be used for sum calculations
        :param fields: The DataFrame fields that will be used for the sum calculations
        """
        values = data[fields].sum().rename('tally')
        self.logger.info(values)
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

        if self.key == 'epochmilli':
            values = self.days[[self.key]].merge(instances, how='inner', left_on=self.key, right_index=True)
        else:
            values = self.days[[self.key, 'epochmilli']].merge(instances, how='inner', left_on=self.key, right_index=True)
            values.drop(columns=[self.key], inplace=True)

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
        instances = pd.concat([quantiles, tallies, nonzero], axis=1)

        sticks = self.sticks(instances)

        return sticks