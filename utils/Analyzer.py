import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from typing import List, Dict, Literal
from scipy.stats import ConstantInputWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from .Visualizer import SplitEffectViewer, ValuesComparation, Correlation_Display, plot_dual_y_axis, BackTest_Display

def __Warn__(message: str) -> str: return "\033[31m" + message + "\033[0m"
def __Alert__(message: str) -> str: return "\033[33m" + message + "\033[0m"
warnings.filterwarnings("ignore", category=ConstantInputWarning)

class Factors_Analyzer:
    def __cal_RankIC__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, step: int = 1) -> float:
        try: return factor_Matrix.shift(step).corrwith(other=return_Matrix, axis=1, method='spearman').mean()
        except Exception as _error: return __Alert__(f"\n由于以下原因 RankIC 无法计算:\n{_error}\n")

    def __cal_IC__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, step: int = 1) -> float:
        try: return factor_Matrix.shift(step).corrwith(other=return_Matrix, axis=1, method='pearson').mean()
        except Exception as _error: return __Alert__(f"\n由于以下原因 IC 无法计算:\n{_error}\n")

    def __cal_ICIR__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, step: int = 1) -> float:
        try:
            IC_Series = factor_Matrix.shift(step).corrwith(other=return_Matrix, axis=1, method='pearson')
            return IC_Series.mean(skipna=True) / IC_Series.std(skipna=True)
        except Exception as _error: return __Alert__(f"\n由于以下原因 ICIR 无法计算:\n{_error}\n")

    def __cal_WinRate__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, step: int = 1) -> float:
        try:
            RankIC_Series = factor_Matrix.shift(step).corrwith(other=return_Matrix, axis=1, method='spearman')
            return np.sum(RankIC_Series < 0) / len(RankIC_Series)
        except Exception as _error: return __Alert__(f"\n由于以下原因 WinRate 无法计算:\n{_error}\n")

    def __cal_LongReturn__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, k: float = 0.1) -> float:
        try:
            return_Series = return_Matrix[factor_Matrix.rank(axis=1, ascending=self.ascending) <= (k*factor_Matrix.shape[1])].mean(axis=1)
            return return_Series.cumsum(skipna=True).iloc[-1]
        except Exception as _error: return __Alert__(f"\n由于以下原因 LongReturn 无法计算:\n{_error}\n")

    def __cal_LongShortReturn__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, k: float = 0.1) -> float:
        try:
            return_Series = return_Matrix[factor_Matrix.rank(axis=1, ascending=not self.ascending) <= (k*factor_Matrix.shape[1])].mean(axis=1)
            return self.__cal_LongReturn__(factor_Matrix, return_Matrix) - return_Series.cumsum(skipna=True).iloc[-1]
        except Exception as _error: return __Alert__(f"\n由于以下原因 LongShortReturn 无法计算:\n{_error}\n")

    def __cal_LongShortSharpe__(self, factor_Matrix: pd.DataFrame, return_Matrix: pd.DataFrame, k: float = 0.1) -> float:
        try:
            long_Series = return_Matrix[factor_Matrix.rank(axis=1, ascending=self.ascending) <= (k*factor_Matrix.shape[1])].mean(axis=1)
            short_Series = return_Matrix[factor_Matrix.rank(axis=1, ascending=not self.ascending) <= (k*factor_Matrix.shape[1])].mean(axis=1)
            return (long_Series-short_Series).mean(skipna=True) / (long_Series-short_Series).std(skipna=True)*np.sqrt(365)
        except Exception as _error: return __Alert__(f"\n由于以下原因 LongShortSharpe 无法计算:\n{_error}\n")
        
    def __wash_and_check_data__(self, data_Matrix: pd.DataFrame) -> pd.DataFrame:
        data_Matrix.sort_index(inplace=True)
        data_Matrix.interpolate(limit_direction='both',inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(data_Matrix.index): raise TypeError("序列索引非时间格式，请检查")
        else: return data_Matrix

    def __check_MatrixS_shapes__(self, data_MatrixS: List[pd.DataFrame]) -> None:
        if len(set([data_Matrix.shape for data_Matrix in data_MatrixS])) > 1:
            raise AssertionError("警告：形状指标必须一致!")

        if len(set(map(tuple,[data_Matrix.index for data_Matrix in data_MatrixS]))) > 1:
            raise AssertionError("警告：所有索引必须一致!")

    def __init__(self, 
                 factor_MatrixS: List[pd.DataFrame], 
                 return_Matrix: pd.DataFrame, 
                 factor_names: List[str] = None,
                 ascending: bool = False) -> None:

        self.factor_MatrixS = [self.__wash_and_check_data__(factor_Matrix) for factor_Matrix in factor_MatrixS]
        self.return_Matrix = self.__wash_and_check_data__(return_Matrix)

        self.__check_MatrixS_shapes__(self.factor_MatrixS + [self.return_Matrix])

        if factor_names != None:
            assert len(factor_names) == len(factor_MatrixS), "警告：因子名称列表长度和因子值矩阵数量必须一致！"
            self.factor_names = factor_names
            self.with_names = True
        else: self.with_names = False

        self.ascending = ascending

    def factor_correlation(self, display:bool = False) -> pd.DataFrame:
        """
        因子相关性分析
        display: 是否显示
        """
        correlation_MatrixS = [pd.concat([factor_Matrix.loc[time_spot] 
                                          for factor_Matrix in self.factor_MatrixS], axis=1, ignore_index=False).corr(method='pearson').fillna(0).values
                               for time_spot in self.return_Matrix.index]

        correlation_Matrix = pd.DataFrame(np.mean(np.stack(correlation_MatrixS, axis=0), axis=0),
                                          index=self.factor_names if self.with_names else [f'Factor_{i+1}' for i in len(self.factor_MatrixS)],
                                          columns=self.factor_names if self.with_names else [f'Factor_{i+1}' for i in len(self.factor_MatrixS)])

        if display: Correlation_Display(correlation_Matrix, mark='Factor Correlation Matrix')
        return correlation_Matrix

    def factor_analyze(self, step: int = 0) -> pd.DataFrame:
        analyzation =  pd.concat([pd.DataFrame({'RankIC'    : self.__cal_RankIC__(factor_Matrix.shift(step).bfill(), self.return_Matrix, step=step),
                                                'IC'        : self.__cal_IC__(factor_Matrix.shift(step).bfill(), self.return_Matrix, step=step),
                                                'ICIR'      : self.__cal_ICIR__(factor_Matrix.shift(step).bfill(), self.return_Matrix, step=step),
                                                '胜率'       : self.__cal_WinRate__(factor_Matrix.shift(step).bfill(), self.return_Matrix),
                                                '多头年化收益': self.__cal_LongReturn__(factor_Matrix.shift(step).bfill(), self.return_Matrix),
                                                '多空年化收益': self.__cal_LongShortReturn__(factor_Matrix.shift(step).bfill(), self.return_Matrix),
                                                '多空年化夏普': self.__cal_LongShortSharpe__(factor_Matrix.shift(step).bfill(), self.return_Matrix)
                                                },index=[0]) for factor_Matrix in self.factor_MatrixS], axis=0, ignore_index=True)
        if self.with_names: analyzation.index = self.factor_names
        return analyzation

    def factor_split_effect(self, step: int = 0, layers: int = 10) -> None:
        layer_size = self.factor_MatrixS[0].shape[1] // layers
        split_MatrixS =  [pd.concat(
            [(self.return_Matrix[(factor_Matrix.shift(step).rank(axis=1, ascending=self.ascending) >= layer*layer_size)&
                                 (factor_Matrix.shift(step).rank(axis=1, ascending=self.ascending) < (layer+1)*layer_size)].mean(axis=1)).cumsum(skipna=True) + 1
             for layer in range(layers)],
             axis=1, join='inner', ignore_index=False).interpolate(axis=0, limit_direction='both')
        for factor_Matrix in self.factor_MatrixS]
        _ = SplitEffectViewer(split_MatrixS, self.factor_names) if self.with_names else SplitEffectViewer(split_MatrixS)
        

    def factor_investment_simulation(self, factor_Matrix: pd.DataFrame, price_Matrix: pd.DataFrame, 
                                     size: int = None, step: int = 0, 
                                     mode: Literal['average', 'hold']='average',
                                     output: Literal['portfolio', 'choice', 'position']='portfolio') -> pd.Series:
        """
        因子模拟交易：

        参数
        - factor_Matrix: 因子矩阵
        - price_Matrix: 典型价格矩阵
        - size: 持仓币种数量 默认为总数的10%
        - mode: 调仓模式【average: 平均调仓, hold: 保持调仓】
        - step: 因子滞后阶数【推荐提前自行滞后保证每个时间点都有信号！！！】
        """
        self.__check_MatrixS_shapes__([factor_Matrix, price_Matrix])
        assert mode in ['average', 'hold'], "警告：调仓模式参数错误！"
        assert output in ['portfolio', 'choice', 'position'], "警告：输出参数错误！"
        if size == None: size = int(price_Matrix.shape[1] * 0.1)
        if step > 0: factor_Matrix.shift(step)

        signal_Matrix = factor_Matrix.rank(axis=1, method='average', ascending=self.ascending) <= size
        choice_Matrix:pd.DataFrame = factor_Matrix.where(signal_Matrix).apply(lambda x: x.dropna().index, axis=1, result_type='expand')
        choice_Matrix.columns = [f'Rank_{i+1}' for i in range(size)]

        # 初始化持仓情况
        pos_Matrix = pd.DataFrame(0, index=price_Matrix.index, columns=price_Matrix.columns, dtype=float) # 置与choice_Matrix的空仓位矩阵

        pos_list = choice_Matrix.loc[choice_Matrix.index[0]].values.tolist() # 调取当期仓位
        pos_Matrix.loc[pos_Matrix.index[0], pos_list] = 1 / price_Matrix.loc[pos_Matrix.index[0], pos_list].values # 输入当期仓位

        if mode == 'average':
            for i, time_spot in enumerate(signal_Matrix.index[1:]):
                # 生成待调仓列表
                new_pos_list = choice_Matrix.loc[time_spot].values.tolist() # 调取当期仓位

                # 平仓
                cash = (price_Matrix.loc[time_spot, pos_list].values * pos_Matrix.loc[signal_Matrix.index[i], pos_list].values).sum()

                # 平均开仓
                cash_distributed = cash / size
                pos_Matrix.loc[time_spot, new_pos_list] = cash_distributed / price_Matrix.loc[time_spot, new_pos_list].values

                pos_list = new_pos_list
                    
        if mode == 'hold':
            for i, time_spot in enumerate(signal_Matrix.index[1:]):
                # 生成待调仓列表
                new_pos_list = choice_Matrix.loc[time_spot].values.tolist() # 调取当期仓位
                to_sell_list = list(set(pos_list) - set(new_pos_list))
                to_buy_list  = list(set(new_pos_list) - set(pos_list))
                to_hold_list = list(set(pos_list) & set(new_pos_list))
                pos_list = new_pos_list

                # 持仓
                pos_Matrix.loc[time_spot, to_hold_list] = pos_Matrix.loc[signal_Matrix.index[i], to_hold_list].values

                # 平仓
                cash = (price_Matrix.loc[time_spot, to_sell_list].values * pos_Matrix.loc[signal_Matrix.index[i], to_sell_list].values).sum()

                # 平均开仓
                if cash > 0:
                    cash_distributed = cash / len(to_buy_list)
                    pos_Matrix.loc[time_spot, to_buy_list] = cash_distributed / price_Matrix.loc[time_spot, to_buy_list].values

        portfolio = (pos_Matrix * price_Matrix).sum(axis=1) / size

        if output == 'portfolio': return portfolio
        if output == 'choice': return choice_Matrix
        if output == 'position': return pos_Matrix

    def factors_investment_test(self, price_Matrix: pd.DataFrame, 
                                size: int, mode: Literal['average', 'hold']='average', step: int = 0,
                                freq: Literal['D', 'H', 'M'] = 'D',
                                visual: bool = False) -> pd.DataFrame:
        """
        因子模拟交易分析：

        参数
        - size: 持仓币种数量 推荐为总数的10%
        - mode: 调仓模式【average: 平均调仓, hold: 保持调仓】
        - step: 因子滞后阶数【推荐提前自行滞后保证每个时间点都有信号！！！】
        - visual: 是否可视化
        """
        portfolio_Matrix = pd.concat([self.factor_investment_simulation(factor_Matrix=factor_Matrix, 
                                                                        price_Matrix=self.__wash_and_check_data__(price_Matrix), 
                                                                        size=size,
                                                                        step=step,
                                                                        mode=mode) 
                                      for factor_Matrix in self.factor_MatrixS],
                                     axis=1, ignore_index=False, join='inner')
        portfolio_Matrix.columns = self.factor_names if self.with_names else [f'Factor_{i+1}' for i in range(len(self.factor_MatrixS))]

        if visual: ValuesComparation(portfolio_Matrix)

        return  pd.DataFrame([Portfolio_Analyzer(portfolio_Matrix[factor_name],freq).analyze_overall() 
                              for factor_name in portfolio_Matrix.columns], index=portfolio_Matrix.columns)

    def execute_RFE_analyze(self, target:int=3):
        X_MatrixS = [factor_Matrix.sub(factor_Matrix.mean(axis=1, skipna=True), axis=0).subtract(factor_Matrix.std(axis=1, skipna=True), axis=0)
                     for factor_Matrix in self.factor_MatrixS]
        X = pd.concat([X_Matrix.stack().reset_index(drop=True) for X_Matrix in X_MatrixS], axis=1)
        X.columns = self.factor_names if self.with_names else [f'factor_{i + 1}' for i in range(len(X_MatrixS))]
        y = (self.return_Matrix > 0).stack().reset_index(drop=True)

        model = RandomForestClassifier()
        rfe = RFE(estimator=model, n_features_to_select=target, step=1)
        rfe.fit(X, y)

        print("Feature Ranking:", rfe.ranking_)

        selected_features = X.columns[rfe.support_]
        print("Selected Features:", selected_features)

    def execute_RFECV_analyze(self):
        X_MatrixS = [factor_Matrix.sub(factor_Matrix.mean(axis=1, skipna=True), axis=0).subtract(factor_Matrix.std(axis=1, skipna=True), axis=0)
                     for factor_Matrix in self.factor_MatrixS]
        X = pd.concat([X_Matrix.ffill().bfill().stack().reset_index(drop=True) for X_Matrix in X_MatrixS], axis=1)
        X.columns = self.factor_names if self.with_names else [f'factor_{i + 1}' for i in range(len(X_MatrixS))]
        y = (self.return_Matrix.ffill().bfill() > 0).stack().reset_index(drop=True)

        model = RandomForestClassifier(random_state=888)
        cv = StratifiedKFold()
        rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy', n_jobs=4)

        rfecv.fit(X,y)

        plt.figure(figsize=(10, 6))
        plt.title('Recursive Feature Elimination with Cross-Validation')
        plt.xlabel('Number of Features Selected')
        plt.ylabel('Cross-Validation score (accuracy)')

        for i in range(cv.n_splits):
            plt.plot(range(1, len(rfecv.cv_results_[f'split{i}_test_score']) + 1),
                     rfecv.cv_results_[f'split{i}_test_score'],
                     marker='o', linestyle='--', label=f'Fold {i+1}')

        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                 rfecv.cv_results_['mean_test_score'], marker='*', color='red', label='Mean CV Accuracy')

        plt.axvline(x=rfecv.n_features_, color='gray', linestyle='--', label=f'Optimal Features:{rfecv.n_features_}')
        plt.legend()
        plt.grid(True)
        plt.show()

        

def get_performance(signal_daily:pd.Series, benchmark:pd.Series, display:bool = True, return_Series:bool = False):
    assert set(signal_daily.unique()).issubset({0, 1}), "Signal must be 0 or 1"
    assert signal_daily.index.equals(benchmark.index), "Signal and benchmark must have the same time stamps"

    portfolio = (benchmark.pct_change().shift(-1).fillna(0) * signal_daily + 1).cumprod()
    returns = portfolio.pct_change().fillna(0)
    DRAWDOWN = (portfolio / portfolio.cummax()).min() - 1
    EXCEED_AR = (portfolio.iloc[-1] / portfolio.iloc[0] - benchmark.iloc[-1] / benchmark.iloc[0] + 1) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
    ANNUALRETURN = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
    SHARPE = returns.mean(skipna=True) / returns.std(skipna=True) * np.sqrt(250)
    try:
        SORTINO = ((returns.mean(skipna=True) / returns[returns < 0].std(skipna=True)) * (np.sqrt(len(returns) / len(returns[returns < 0]))) * np.sqrt(250))
    except Exception as e:
        tqdm.write('e')
        SORTINO = np.nan
    CARMAR = (ANNUALRETURN / DRAWDOWN * -1) if DRAWDOWN != 0 else np.nan

    # Significance Analyzation
    origin = sm.add_constant(benchmark)
    model = sm.OLS(portfolio.astype(float), origin).fit()
    flag = 1 if model.params['const'] > 0 else -1
    ALPHA_PVALUE = model.pvalues['const'] * flag

    # Signal Analyzation
    signal_daily = signal_daily.diff().fillna(0)

    signal = ((signal_daily > 0) * 1 + (signal_daily < 0) * (-1)).replace(0, np.NaN).dropna()
    bench_value = benchmark
    buy_signals = signal.loc[signal == 1]
    buy_dates = buy_signals.index.tolist()
    # 买入信号日期
    if not buy_dates:
        sell_dates = []
    else:
        # 获取最早的买入日期
        first_buy_date = buy_dates[0]
        # 提取卖出信号日期，确保卖出日期在最早的买入日期之后
        sell_signals = signal.loc[(signal == -1) & (signal.index > first_buy_date)]
        sell_dates = sell_signals.index.tolist()  # 卖出信号日期列表
    if len(sell_dates) < len(buy_dates):
        sell_dates += [bench_value.index[-1]]

    order_num = len(buy_dates)
    if order_num > 2:
        order_ret_buy = pd.Series([bench_value.loc[sell_dates[i]] / bench_value.loc[buy_dates[i]] - 1 for i in range(order_num)])
        order_ret_short = pd.Series([bench_value.loc[buy_dates[i + 1]] / bench_value.loc[sell_dates[i]] - 1 for i in range(order_num - 1)])
        WINRATE = (order_ret_buy > 0).sum() / len(order_ret_buy)
        RWINRATE = (order_ret_buy[order_ret_buy > 0]).sum() / order_ret_buy.abs().sum()
        LOSSRISK = (order_ret_buy[order_ret_buy > 0]).mean() / (order_ret_buy[order_ret_buy < 0]).mean() * -1

        SHORTWINRATE = (order_ret_short < 0).sum() / len(order_ret_short)
        SHORTRWINRATE = abs((order_ret_short[order_ret_short < 0]).sum()) / order_ret_short.abs().sum()
        SHORTLOSSRISK = (order_ret_short[order_ret_short < 0]).mean() / (order_ret_short[order_ret_short > 0]).mean() * -1
    else:
        warnings.warn("No valid buy orders found. Please check the signal data.")
        WINRATE = np.nan
        RWINRATE = np.nan
        LOSSRISK = np.nan

        SHORTWINRATE = np.nan
        SHORTRWINRATE = np.nan
        SHORTLOSSRISK = np.nan

    if display:
        print(f'Orders {order_num}')
        print(f'Drawdown {DRAWDOWN:.4%}')
        print(f'EXCEED AR {EXCEED_AR:.4%}')
        print(f'Annual Return {ANNUALRETURN:.4%}')
        print(f'Sharpe Ratio {SHARPE:.4f}')
        print(f'Sortino Ratio {SORTINO:.4f}')
        print(f'Carmar Ratio {CARMAR:.4f}')
        print(f'Win Rate {WINRATE:.4%}')
        print(f'Short Win Rate {WINRATE:.4%}')
        print(f'Return Weighted Win Rate {RWINRATE:.4%}')
        print(f'Short Return Weighted Win Rate {SHORTRWINRATE:.4%}')
        print(f'Loss Risk {LOSSRISK:.4f}')
        print(f'Short Loss Risk {SHORTLOSSRISK:.4f}')
        print(f'Excpected Return {(WINRATE * LOSSRISK):.4f}')    
        print(f'Excpected Short Return {(SHORTWINRATE * SHORTLOSSRISK):.4f}')

        plot_data = pd.concat([benchmark, portfolio], axis=1)
        plot_data = plot_data / plot_data.iloc[0]  # Normalize to the first value
        plot_data.columns = ['price', 'portfolio']
        BackTest_Display(plot_data, 'portfolio')

    if return_Series:
        return pd.Series(
            [f'{portfolio.index[0]}-{portfolio.index[-1]}', DRAWDOWN, ANNUALRETURN, SHARPE, SORTINO, CARMAR, EXCEED_AR, WINRATE, RWINRATE, LOSSRISK, WINRATE * LOSSRISK],
            index=['Period', 'DrawDown', 'AnnualReturn', 'Sharpe', 'Sortino', 'Carmar', 'Excessive AR', 'WinRate', 'RWinRate', 'lossRisk', 'ExpectedReturn'])
    else:   
        return DRAWDOWN, ANNUALRETURN, SHARPE, SORTINO, CARMAR, EXCEED_AR, WINRATE, RWINRATE, LOSSRISK, WINRATE * LOSSRISK  


def get_soft_performance(
    position_daily: pd.Series, 
    benchmark: pd.Series, 
    fee_rate: float = 0,
    soft_range: float = 0.1, 
    display: bool = True
):
    assert np.sum((position_daily >= 0) & (position_daily <= 1)) == len(position_daily), "Signal must between 0 and 1"
    assert position_daily.index.equals(benchmark.index), "Signal and benchmark must have the same time stamps"

    cash = 1
    shares = 0
    position = 0
    portfolio = []
    for index in position_daily.index:

        if position_daily.loc[index] < 0.5 - soft_range:
            cash += shares * benchmark.loc[index] * (1 - fee_rate)
            shares = 0
            position = 0

        elif (0.5 - soft_range) <= position_daily.loc[index] <= (0.5 + soft_range):
            new_position = position_daily.loc[index]

            if position == 0:
                shares = cash * new_position / benchmark.loc[index]
                cash = cash * (1 - new_position) - cash * new_position * fee_rate

            elif position == 1:
                cash = (1 - new_position) * shares * benchmark.loc[index] * (1 - fee_rate)
                shares = shares * new_position

            else:
                delta_position = new_position - position
                shares += cash * delta_position / (1 - position) / benchmark.loc[index]
                cash = cash * (1 - new_position) / (1 - position) - abs(cash * delta_position / (1 - position) * fee_rate)

            position = new_position

        elif position_daily.loc[index] > 0.5 + soft_range:
            shares += cash / benchmark.loc[index] * (1 - fee_rate)
            cash = 0
            position = 1

        portfolio.append(cash + shares * benchmark.loc[index])

    portfolio = pd.Series(portfolio, position_daily.index)

    returns = portfolio.pct_change().fillna(0)
    DRAWDOWN = (portfolio / portfolio.cummax()).min() - 1
    EXCEED_AR = (portfolio.iloc[-1] / portfolio.iloc[0] - benchmark.iloc[-1] / benchmark.iloc[0] + 1) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
    ANNUALRETURN = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
    SHARPE = returns.mean(skipna=True) / returns.std(skipna=True) * np.sqrt(250)
    try:
        SORTINO = ((returns.mean(skipna=True) / returns[returns < 0].std(skipna=True)) * (np.sqrt(len(returns) / len(returns[returns < 0]))) * np.sqrt(250))
    except Exception as e:
        tqdm.write('e')
        SORTINO = np.nan
    CARMAR = (ANNUALRETURN / DRAWDOWN * -1) if DRAWDOWN != 0 else np.nan

    if display:
        print(f'Drawdown {DRAWDOWN:.4%}')
        print(f'EXCEED AR {EXCEED_AR:.4%}')
        print(f'Annual Return {ANNUALRETURN:.4%}')
        print(f'Sharpe Ratio {SHARPE:.4f}')
        print(f'Sortino Ratio {SORTINO:.4f}')
        print(f'Carmar Ratio {CARMAR:.4f}')

        plot_data = pd.concat([benchmark, portfolio], axis=1)
        plot_data = plot_data / plot_data.iloc[0]  # Normalize to the first value
        plot_data.columns = ['price', 'portfolio']
        plot_data.to_csv('observe.csv')
        BackTest_Display(plot_data, 'portfolio')

    return pd.Series(
            [f'{portfolio.index[0]}-{portfolio.index[-1]}', DRAWDOWN, ANNUALRETURN, SHARPE, SORTINO, CARMAR, EXCEED_AR],
            index=['Period', 'DrawDown', 'AnnualReturn', 'Sharpe', 'Sortino', 'Carmar', 'Excessive AR'])

def get_soft_performance_2(
    position_daily: pd.Series, 
    benchmark: pd.Series, 
    fee_rate: float = 0,
    soft_range: float = 0.1, 
    display: bool = True
):
    assert ((position_daily >= 0) & (position_daily <= 1)).all(), "Signal must between 0 and 1"
    assert position_daily.index.equals(benchmark.index), "Signal and benchmark must have the same time stamps"

    cash = 1
    shares = 0
    position = 0
    portfolio = []
    for index in position_daily.index:

        if position_daily.loc[index] < 0.5 - soft_range:
            cash += shares * benchmark.loc[index] * (1 - fee_rate)
            shares = 0
            position = 0

        elif (0.5 - soft_range) <= position_daily.loc[index] <= (0.5 + soft_range):
            new_position = 0.5

            if position == 0:
                shares = cash * new_position / benchmark.loc[index]
                cash = cash * (1 - new_position) - cash * new_position * fee_rate

            elif position == 1:
                cash = (1 - new_position) * shares * benchmark.loc[index] * (1 - fee_rate)
                shares = shares * new_position

            position = new_position

        elif position_daily.loc[index] > 0.5 + soft_range:
            shares += cash / benchmark.loc[index] * (1 - fee_rate)
            cash = 0
            position = 1

        portfolio.append(cash + shares * benchmark.loc[index])

    portfolio = pd.Series(portfolio, position_daily.index)

    returns = portfolio.pct_change().fillna(0)
    DRAWDOWN = (portfolio / portfolio.cummax()).min() - 1
    EXCEED_AR = (portfolio.iloc[-1] / portfolio.iloc[0] - benchmark.iloc[-1] / benchmark.iloc[0] + 1) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
    ANNUALRETURN = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
    SHARPE = returns.mean(skipna=True) / returns.std(skipna=True) * np.sqrt(250)
    try:
        SORTINO = ((returns.mean(skipna=True) / returns[returns < 0].std(skipna=True)) * (np.sqrt(len(returns) / len(returns[returns < 0]))) * np.sqrt(250))
    except Exception as e:
        tqdm.write('e')
        SORTINO = np.nan
    CARMAR = (ANNUALRETURN / DRAWDOWN * -1) if DRAWDOWN != 0 else np.nan

    if display:
        print(f'Period:{portfolio.index[0]}-{portfolio.index[-1]}')
        print(f'Drawdown {DRAWDOWN:.4%}')
        print(f'EXCEED AR {EXCEED_AR:.4%}')
        print(f'Annual Return {ANNUALRETURN:.4%}')
        print(f'Sharpe Ratio {SHARPE:.4f}')
        print(f'Sortino Ratio {SORTINO:.4f}')
        print(f'Carmar Ratio {CARMAR:.4f}')

        plot_data = pd.concat([benchmark, portfolio], axis=1)
        plot_data = plot_data / plot_data.iloc[0]  # Normalize to the first value
        plot_data.columns = ['price', 'portfolio']
        plot_data.to_csv('observe.csv')
        BackTest_Display(plot_data, 'portfolio')

    return pd.Series(
            [f'{portfolio.index[0]}-{portfolio.index[-1]}', DRAWDOWN, ANNUALRETURN, SHARPE, SORTINO, CARMAR, EXCEED_AR],
            index=['Period', 'DrawDown', 'AnnualReturn', 'Sharpe', 'Sortino', 'Carmar', 'Excessive AR'])


class Portfolio_Analyzer:
    """"""
    def __cal_Drawdown__(self, portfolio: pd.Series):
        """最大回撤"""
        try:
            return 1 - (portfolio / portfolio.cummax()).min()
        except Exception as _error:
            return __Alert__(f"\n由于以下原因 最大回撤 无法计算:\n{_error}\n")

    def __cal_Annual_Return__(self, portfolio: pd.Series):
        """年化收益"""
        try:
            return (portfolio.loc[portfolio.index[-1]] / portfolio.loc[portfolio.index[0]]) ** (360 / (portfolio.index[-1] - portfolio.index[0]).days) - 1
        except Exception as _error:
            return __Alert__(f"\n由于以下原因 年化收益 无法计算:\n{_error}\n")

    def __cal_Sharpe__(self, portfolio: pd.Series):
        """夏普比率"""
        try:
            returns: pd.Series = portfolio.pct_change().dropna()
            return (returns.mean(skipna=True) - self.rf / (250 * self.T)) / returns.std(skipna=True) * np.sqrt(250 * self.T)
        except Exception as _error:
            return __Alert__(f"\n由于以下原因 夏普比率 无法计算:\n{_error}\n")

    def __cal_Sortino__(self, portfolio: pd.Series):
        """所提诺比率"""
        try:
            returns: pd.Series = portfolio.pct_change().dropna()
            return (((returns.mean(skipna=True) - self.rf / (250 * self.T)) / returns[returns < 0].std(skipna=True)) * (np.sqrt(len(returns) / len(returns[returns < 0]))) * np.sqrt(250 * self.T))
        except Exception as _error:
            return __Alert__(f"\n由于以下原因 所提诺比率 无法计算\n{_error}\n")

    def __cal_Calmar__(self, portfolio: pd.Series):
        """卡玛比率"""
        try:
            return self.__cal_Annual_Return__(portfolio) / self.__cal_Drawdown__(portfolio)
        except Exception as _error:
            return __Alert__(f"\n由于以下原因 卡玛比率 无法计算\n{_error}\n")

    def __init__(self, 
                 portfolio: pd.Series, 
                 rf: float,
                 freq: Literal['Q', 'ME', 'D', 'H', 'M'] = 'D') -> None:
        if not pd.api.types.is_datetime64_any_dtype(portfolio.index): raise TypeError("序列索引非时间格式，请检查")
        else: self.portfolio = portfolio

        self.rf = rf

        if freq not in ['Q', 'ME', 'D', 'H', 'M']: raise ValueError("频率参数错误，请检查")
        else: self.T = {'Q': 3/250, 'ME': 12/250, 'D': 1, 'H': 24, 'M': 1440}[freq]
        

    def __analyze__(self, portfolio:pd.Series) -> Dict[str, float|str]:
        return {'最大回撤'  : self.__cal_Drawdown__(portfolio),
                '年化收益率': self.__cal_Annual_Return__(portfolio),
                '夏普比率'  : self.__cal_Sharpe__(portfolio),
                '所提诺比率': self.__cal_Sortino__(portfolio),
                '卡玛比率'  : self.__cal_Calmar__(portfolio)}

    def analyze_overall(self, display: bool = False) -> Dict[str, float|str]:
        """
        全局数据 经济学效应测试
        参数：
        - display: 是否显示
        """
        test_result = self.__analyze__(self.portfolio)

        if any(isinstance(value, str) for value in test_result.values()):
            print(__Warn__('警告！计算中出现了错误，请查看并检查数据！'))

        if display:
            for key in test_result.keys(): print(f'{key}: {test_result[key]}')

        return test_result

    def analyze_yearly(self, display: bool = False) -> pd.DataFrame:
        """
        分年数据 经济学效应测试
        参数：
        - display: 是否显示
        """
        temp_portfolio = self.portfolio.to_frame(name='value')
        temp_portfolio['year'] = temp_portfolio.index.year
        result = temp_portfolio.groupby(by='year')['value'].apply(lambda _portfolio: pd.Series(self.__analyze__(_portfolio))).unstack()

        if display: print(result)

        return result