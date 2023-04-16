import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm
import os
from scipy.optimize import curve_fit
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
class OptionImpliedDist:
    @staticmethod
    def get_implied_prob_array(call_curve, tau: float, points: np.ndarray, delta = None, r = 0.01):
        if delta is None:
            delta = min(np.median(points)*0.01, 0.01)

        points_bellow = points - delta
        points_above = points + delta
        call_bellow = np.maximum(call_curve(points_bellow), 10**-4)
        call_above = np.maximum(call_curve(points_above), 10**-4)
        call = np.maximum(call_curve(points), 10**-4)
        C2K2 = (call_bellow + call_above - 2*call) / (delta**2)
        C2K2 = np.abs(C2K2)
        density = C2K2*np.exp(r*tau)
        density_ser = pd.Series(density, index=points)
        return density_ser
    @staticmethod
    def fit_normal_dist(density_ser: pd.Series):
        mean = np.mean(density_ser.index.values)
        std = np.std(density_ser.index.values)
        popt = curve_fit(norm.pdf, density_ser.index.values, density_ser.values, p0=[mean, std], bounds=(0, np.inf))
        mu, sigma = popt[0]
        return mu, sigma

    @staticmethod
    def fit_log_normal(density_ser: pd.Series):
        mean = np.mean(density_ser.index.values)
        std = np.std(density_ser.index.values)
        fun = lambda x, mean, std: lognorm.pdf(x, s=1, loc=mean, scale=std)
        popt = curve_fit(fun, density_ser.index.values, density_ser.values, p0=[mean, std], bounds=(0, np.inf))
        mu, sigma = popt[0]
        return mu, sigma

    @staticmethod
    def preprocess_data(symbol):
        filename = f"call_curves_{symbol}.pickle"
        filepath = os.path.join("../data/", filename)
        with open(filepath, 'rb') as handle:
            curve_dict = pickle.load(handle)
        today = datetime.now()
        format_dt = '%B %d, %Y'
        # calculate tau
        res = {}
        for date, data in curve_dict.items():
            dt = datetime.strptime(date, format_dt)
            tau = (dt - today).total_seconds() / (365.25*24*60*60)
            res[tau] = data

        return res

    @staticmethod
    def plot_results(density_ser, **kwargs):
        mu = kwargs["mu"]
        sigma = kwargs["sigma"]
        dist = kwargs['dist']
        if dist == 'norm':
            fitted_values = norm.pdf(density_ser.index, mu, sigma)
        elif dist == 'lognorm':
            fitted_values = lognorm.pdf(density_ser.index, mu, sigma)
        else:
            raise ValueError('distribution not implemented')
        fitted_ser = pd.Series(fitted_values, index = density_ser.index)
        density_df = density_ser.to_frame()
        density_df.columns = ['actual']
        density_df['fitted'] = fitted_ser
        fig, ax = plt.subplots()
        density_df.plot(ax= ax)
        fig.show()



    @staticmethod
    def main(symbol, dist = "lognorm"):
        res = OptionImpliedDist.preprocess_data(symbol)
        for tau, (curve, df) in res.items():
            density_ser = OptionImpliedDist.get_implied_prob_array(curve, tau, points=df['Strike'].values, delta = None, r = 0.01)
            if dist == 'norm':
                mu, sigma = OptionImpliedDist.fit_normal_dist(density_ser)
            elif dist == 'lognorm':
                mu, sigma = OptionImpliedDist.fit_log_normal(density_ser)
            else:
                raise ValueError('distribution not implemented')
            OptionImpliedDist.plot_results(density_ser, mu=mu, sigma=sigma, dist=dist)
            print(f"tau = {tau} ; mu = {mu} ; sigma = {sigma}")


if __name__ == '__main__':
    symbol = 'AAPL'
    OptionImpliedDist.main(symbol, dist='norm')



