from yahoo_fin import options
from scipy.interpolate import CubicSpline
import pickle
import os
class OptionProvider:
    @staticmethod
    def get_option_chain(symbol):
        exp_dates = options.get_expiration_dates(symbol)
        info = {}
        for date in exp_dates:
            info[date] = options.get_options_chain(symbol, date)
        return info

    @staticmethod
    def get_calls(symbol):
        exp_dates = options.get_expiration_dates(symbol)
        info = {}
        for date in exp_dates:
            info[date] = options.get_calls(symbol, date)
            info[date].sort_values('Strike')
        return info
    @staticmethod
    def get_option_curve(strike_ser, price_ser):
        fun = CubicSpline(strike_ser, price_ser)
        return fun

    @staticmethod
    def get_call_data(symbol:str):
        call_chain = OptionProvider.get_calls(symbol)
        res = {}
        for date, df in call_chain.items():
            curve = OptionProvider.get_option_curve(df['Strike'], df['Last Price'])
            res[date] = (curve, df)
        return res

    @staticmethod
    def get_call_main(symbol):
        call_res = OptionProvider.get_call_data(symbol)
        filename = f"call_curves_{symbol}.pickle"
        filepath = os.path.join("../data/", filename)
        with open(filepath, 'wb') as handle:
            pickle.dump(call_res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    symbol = 'SPY'
    OptionProvider.get_call_main(symbol)



