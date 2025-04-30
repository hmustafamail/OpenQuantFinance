# Collection of sophisticated volatility estimation methods. 
# These functions yielded plausible results on historical data, 
# but they have not been validated against standard 
# implementations.
#
# Created: April 30, 2025
#
# Python Version: 3.12
#
# Dependencies: pandas, numpy
#
# Author: Mustafa Hussain <h.mustafa.mail@gmail.com>
#
# License: CC BY-SA https://creativecommons.org/licenses/by-sa/4.0/
#
# You are free to share and adapt as long as you give appropriate 
# credit to the author and any derivatives of this work are shared 
# under the same license as the original.
#
# Disclaimer: The software is provided "as is", without warranty 
# of any kind, express or implied, including but not limited 
# to the warranties of merchantability, fitness for a particular 
# purpose and noninfringement. In no event shall the authors or 
# copyright holders be liable for any claim, damages or other 
# liability, whether in an action of contract, tort or otherwise, 
# arising from, out of or in connection with the software or the 
# use or other dealings in the software.


import numpy as np


def parkinson_volatility(high, low, window=10):
    """
    Computes rolling Parkinson volatility.

    Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. Journal of business, 61-65.

    Parameters:
    - high, low (pd.Series): High and low price series
    - window (int): rolling window length

    Returns:
    - pd.Series: Parkinson volatility (not annualized)
    """

    # Constant
    factor = 1 / (4 * np.log(2))

    # Squared log range
    hl_squared = np.log(high / low) ** 2

    # Return rolling standard deviation
    return np.sqrt(factor * hl_squared.rolling(window).mean())


def garman_klass_volatility(open_, high, low, close, window=10):
    """
    Computes Garman-Klass volatility over a rolling window.

    Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. Journal of business, 67-78.

    Parameters:
    - open_, high, low, close (pd.Series): OHLC prices
    - window (int): rolling window length

    Returns:
    - pd.Series: Garman-Klass volatility (not annualized)
    """

    # High-low term
    hl_term = np.log(high / low) ** 2

    # Close-open term
    co_term = np.log(close / open_) ** 2

    # Close-open coefficient
    coef = (2 * np.log(2)) - 1

    # Compute variance
    variance = (0.5 * hl_term) - (coef * co_term)

    # Standard deviation
    return np.sqrt(variance.rolling(window).mean())


def rogers_satchell_volatility(open_, high, low, close, window=10):
    """
    Computes Rogers-Satchell volatility over a rolling window.

    Rogers, L. C. G., & Satchell, S. E. (1991). Estimating variance from high, low and closing prices. The Annals of Applied Probability, 504-512.

    Parameters:
    - open_, high, low, close (pd.Series): OHLC price series
    - window (int): Rolling window length

    Returns:
    - pd.Series: Rolling Rogers-Satchell volatility (not annualized)
    """

    # High-open series
    u = np.log(high) - np.log(open_)

    # Low-open series
    d = np.log(low) - np.log(open_)

    # Close-open series
    c = np.log(close) - np.log(open_)

    # The Rogers Satchell series
    rs = u * (u - c) + d * (d - c)

    # The standard deviation series is the root of the rolling mean
    return np.sqrt(rs.rolling(window).mean())


def garman_klass_yang_zhang_volatility(open_, high, low, close, window=10):
    """
    Computes Garman-Klass-Yang-Zhang volatility over a rolling window.

    Yang, D., & Zhang, Q. (2000). Drift‐independent volatility estimation based on high, low, open, and close prices. The Journal of Business, 73(3), 477-492.

    Parameters:
    - open_, high, low, close (pd.Series): OHLC prices
    - window (int): rolling window length

    Returns:
    - pd.Series: GKYZ volatility (not annualized)
    """

    # Gap term
    gap_sq = np.log(open_ / close.shift(1)) ** 2

    # High-low term
    hl_sq = np.log(high / low) ** 2

    # Close-open term
    co_sq = np.log(close / open_) ** 2

    # Close-open coefficient
    co_coef = (2 * np.log(2)) - 1

    # Per-bar variance
    per_bar_var = gap_sq + 0.5 * hl_sq - co_coef * co_sq

    # To take the rolling variance, take the rolling mean of per-bar variance estimates
    rolling_variance = per_bar_var.rolling(window).mean()

    # Convert to standard deviation
    return np.sqrt(rolling_variance)


def yang_zhang_volatility(open_, high, low, close, window, alpha=1.34):
    """
    Computes rolling Yang-Zhang volatility.

    Yang, D., & Zhang, Q. (2000). Drift‐independent volatility estimation based on high, low, open, and close prices. The Journal of Business, 73(3), 477-492.

    Parameters:
    - open_, high, low, close (pd.Series): OHLC
    - window (int): rolling window length
    - alpha (float): Drift-sensitive weight parameter

    Returns:
    - pd.Series: Yang-Zhang volatility (not annualized)
    """

    # Gap series
    o_s = np.log(open_) - np.log(close.shift(1))

    # Close-open series
    c_s = np.log(close) - np.log(open_)

    # Rolling variance of gap series
    v_o = o_s.rolling(window).var()

    # Rolling variance of close-open series
    v_c = c_s.rolling(window).var()

    # Rogers-Satchell volatility
    v_rs = rogers_satchell_volatility(
        open_, high, low, close, window) ** 2

    # Weighting factor for drift
    k = (alpha - 1) / (alpha + (window + 1) / (window - 1))

    # Construct Yang-Zhang variance by combining weighted components
    yz_var = v_o + (k * v_c) + ((1-k) * v_rs)

    # Return standard deviation
    return np.sqrt(yz_var)


def volatility_mux(candles_dataframe, window=10, method="yang-zhang"):
    """
    Volatility Multiplexer.

    Args:
    - candles_dataframe: A Pandas DataFrame with columns "open", "high", "low", and "close".
    - window: Period of volatility estimation.
    - method: Volatility estimation method. Options: "parkinson", "garman-klass", "rogers-satchell", "garman-klass-yang-zhang", "yang-zhang".

    Methods are referenced by the following paper:

    Bennett, C., & Gil, M. A. (2012). Measuring Historical Volatility. Santander.

    Where the formulae offered by Bennett and Gil appeared to differ from the original works, I did my best to faithfully implement the original formulae.

    Returns a Pandas series of volatility estimates using the specified method (not annualized).
    """

    # Required columns
    required_cols = {"open", "high", "low", "close"}

    # Validate input DataFrame
    if not required_cols.issubset(candles_dataframe.columns):

        # Raise error.
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    match method:

        case "parkinson":

            return parkinson_volatility(
                high=candles_dataframe["high"],
                low=candles_dataframe["low"],
                window=window,
            )

        case "garman-klass":

            return garman_klass_volatility(
                open_=candles_dataframe["open"],
                high=candles_dataframe["high"],
                low=candles_dataframe["low"],
                close=candles_dataframe["close"],
                window=window,
            )

        case "rogers-satchell":

            return rogers_satchell_volatility(
                open_=candles_dataframe["open"],
                high=candles_dataframe["high"], 
                low=candles_dataframe["low"],
                close=candles_dataframe["close"],
                window=window,
            )

        case "garman-klass-yang-zhang":

            return garman_klass_yang_zhang_volatility(
                open_=candles_dataframe["open"],
                high=candles_dataframe["high"],
                low=candles_dataframe["low"],
                close=candles_dataframe["close"],
                window=window,
            )

        case "yang-zhang":

            return yang_zhang_volatility(
                open_=candles_dataframe["open"],
                high=candles_dataframe["high"],
                low=candles_dataframe["low"],
                close=candles_dataframe["close"],
                window=window,
            )

        case _:
            raise ValueError(f"Method {method} not recognized.")
