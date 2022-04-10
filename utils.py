import win32com.client
import pickle
from datetime import datetime, timedelta
import numpy as np


def stock_code_name():
    """종목의 code와 이름 매핑하는 dictionary 저장"""
    # 연결 여부 체크
    objCpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
    bConnect = objCpCybos.IsConnect
    if (bConnect == 0):
        print("PLUS가 정상적으로 연결되지 않음. ")
        exit()

    code_name_dict = {}
    name_code_dict = {}

    def code_name_load():
        objCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
        codeList = objCpCodeMgr.GetStockListByMarket(1)  # 거래소
        codeList2 = objCpCodeMgr.GetStockListByMarket(2)  # 코스닥

        for code in codeList + codeList2:
            name = objCpCodeMgr.CodeToName(code)
            code_name_dict[code] = name
            name_code_dict[name] = code

    code_name_load()

    with open('data/stock_code_name.pickle', 'wb') as f:
        pickle.dump(code_name_dict, f)
        pickle.dump(name_code_dict, f)


def stock_code_market():
    """kospi, kosdaq 종목 리스트 각각 저장"""
    # 연결 여부 체크
    objCpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
    bConnect = objCpCybos.IsConnect
    if (bConnect == 0):
        print("PLUS가 정상적으로 연결되지 않음. ")
        exit()

    objCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
    code_list_kospi = objCpCodeMgr.GetStockListByMarket(1)  # 거래소
    code_list_kosdaq = objCpCodeMgr.GetStockListByMarket(2)  # 코스닥

    with open('data/stock_code_market.pickle', 'wb') as f:
        pickle.dump((code_list_kospi, code_list_kosdaq), f)


def set_datetime_as_index(df, drop=False):
    df = df.copy()
    df.index = df.date.apply(lambda x: datetime.strptime(str(x), '%Y%m%d')) + \
               df.time.apply(lambda x: timedelta(hours=x // 100, minutes=x % 100))
    if drop:
        df.drop(['date', 'time'], axis=1, inplace=True)
    return df


def geometric_avg_er(er):
    # er should be real number, not percentage. output also be real number.
    a = np.array(er) + 1
    return a.prod() ** (1.0 / len(a)) - 1


def price_unit(current_price, code):
    with open('./data/stock_code_market.pickle', 'rb') as fr:
        kospi_list, kosdaq_list = pickle.load(fr)

    # 호가 최소단위
    if current_price < 1000:
        unit = 1
    elif current_price < 5000:
        unit = 5
    elif current_price < 10000:
        unit = 10
    elif current_price < 50000:
        unit = 50
    elif current_price < 100000:
        unit = 100
    elif current_price < 500000:
        if code in kospi_list:
            unit = 500
        else:
            unit = 100
    else:
        if code in kospi_list:
            unit = 1000
        else:
            unit = 100

    return unit


if __name__ == '__main__':
    # stock_code_market()
    price_unit(90000, 'A000660')