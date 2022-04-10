import win32com.client
from datetime import datetime, timedelta
import time
import pandas as pd
from tqdm import tqdm


# 아래 reference 참고 및 변형
# https://github.com/gyusu/Creon-Datareader/blob/master/creonAPI.py


g_objCpStatus = win32com.client.Dispatch('CpUtil.CpCybos')

# original_func 콜하기 전에 PLUS 연결 상태 체크하는 데코레이터
def check_PLUS_status(original_func):

    def wrapper(*args, **kwargs):
        bConnect = g_objCpStatus.IsConnect
        if (bConnect == 0):
            print("PLUS가 정상적으로 연결되지 않음.")
            exit()

        return original_func(*args, **kwargs)

    return wrapper


class CpStockChart:
    def __init__(self):
        """과거의 차트 데이터 가져오는 클래스"""
        self.objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")


    def _check_rq_status(self):
        """통신상태 검사"""
        rqStatus = self.objStockChart.GetDibStatus()
        rqRet = self.objStockChart.GetDibMsg1()
        if rqStatus == 0:
            pass
        else:
            print("통신상태 오류[{}]{} 종료합니다..".format(rqStatus, rqRet))
            exit()


    # 차트 요청 - 최근일 부터 개수 기준
    @check_PLUS_status
    def request_dwm(self, code, dwm, from_date, to_date, ohlcv_only=True, desired_features=None):
        """
        :param code: 종목코드
        :param dwm: 'D':일봉, 'W':주봉, 'M':월봉
        :param from_date: 요청시작일
        :param to_date: 요청종료일
        :param ohlcv_only: True일 경우 ohlcv만 요청
        :param desired_features: 원하는 feature가 있을 경우 list로 전달
        """

        # 주말 및 공휴일 포함하여 count 여유있게 산정
        count = (datetime.strptime(to_date, '%Y%m%d') - datetime.strptime(from_date, '%Y%m%d')).days + 1
        if dwm == 'D':
            pass
        elif dwm == 'W':
            count /= 7
        elif dwm == 'M':
            count /= 28

        self.objStockChart.SetInputValue(0, code)  # 종목코드
        self.objStockChart.SetInputValue(1, ord('2'))  # 개수로 받기
        self.objStockChart.SetInputValue(2, to_date)  # 요청종료일
        self.objStockChart.SetInputValue(4, count)  # 최근 count개

        request_list = [(0, 'date')]
        if desired_features:
            feature_category = [(2, 'open'), (3, 'high'), (4, 'low'), (5, 'close'), (8, 'volume'),
                                (9, '거래대금'), (12, '상장주식수'), (14, '외국인주문한도수량'), (16, '외국인현보유수량'),
                                (17, '외국인현보유비율'), (20, '기관순매수'), (21, '기관누적순매수')]

            assert not [x for x in desired_features if x not in [x[1] for x in feature_category]], \
                f'all features should be in {[x[1] for x in feature_category]}.'

            request_list.extend([
                [cat for cat in feature_category if cat[1] == x][0]
                for x in desired_features
            ])
        else:
            request_list.extend([(2, 'open'), (3, 'high'), (4, 'low'), (5, 'close'), (8, 'volume')])
            if not ohlcv_only:
                request_list.extend([(9, '거래대금'), (12, '상장주식수'), (14, '외국인주문한도수량'), (16, '외국인현보유수량'),
                                     (17, '외국인현보유비율'), (20, '기관순매수'), (21, '기관누적순매수')])
        request_numbers = [x[0] for x in request_list]
        request_columns = [x[1] for x in request_list]

        self.objStockChart.SetInputValue(5, request_numbers)
        self.objStockChart.SetInputValue(6, ord(dwm))  # '차트 주기 - 일/주/월
        self.objStockChart.SetInputValue(9, ord('1'))  # 수정주가 사용

        received_data = {col: [] for col in request_columns}

        while True:
            self.objStockChart.BlockRequest()  # 요청! 후 응답 대기
            self._check_rq_status()  # 통신상태 검사
            time.sleep(0.1)  # 시간당 RQ 제한으로 인해 장애가 발생하지 않도록 딜레이를 줌

            batch_len = self.objStockChart.GetHeaderValue(3)  # 받아온 데이터 개수
            if batch_len == 0:  # 데이터가 없는 경우
                print('데이터 없음')
                return False

            for i in range(batch_len):
                for col_idx, col in enumerate(request_columns):
                    value = self.objStockChart.GetDataValue(col_idx, i)
                    received_data[col].append(value)

            received_oldest_date = min(received_data['date'])
            if received_oldest_date <= int(from_date):
                break

        received_df = pd.DataFrame(received_data)
        received_df = received_df[received_df['date'] >= int(from_date)]
        return received_df.sort_values(['date']).reset_index(drop=True)


    # 차트 요청 - 분 차트
    @check_PLUS_status
    def request_minute(self, code, minute_range, from_date, to_date):
        """
        :param code: 종목코드
        :param minute_range: 1분봉 or 5분봉, ...
        :param from_date: 요청시작일
        :param to_date: 요청종료일
        """

        # 9:01~15:20 380 rows + 15:30 1 row 포함 → 381 rows (5분 간격: 77 rows)
        # 주말 및 공휴일 포함하여 count 여유있게 산정
        count = (datetime.strptime(to_date, '%Y%m%d') - datetime.strptime(from_date, '%Y%m%d')).days
        if minute_range == 1:
            count *= 381
        elif minute_range == 5:
            count *= 77

        self.objStockChart.SetInputValue(0, code)  # 종목코드
        self.objStockChart.SetInputValue(1, ord('2'))  # 개수로 받기
        self.objStockChart.SetInputValue(2, to_date)  # 요청종료일
        self.objStockChart.SetInputValue(4, count)  # 조회 개수

        request_list = [(0, 'date'), (1, 'time'),
                        (2, 'open'), (3, 'high'), (4, 'low'), (5, 'close'), (8, 'volume')]
        request_numbers = [x[0] for x in request_list]
        request_columns = [x[1] for x in request_list]

        self.objStockChart.SetInputValue(5, request_numbers)
        self.objStockChart.SetInputValue(6, ord('m'))  # '차트 주기 - 분
        self.objStockChart.SetInputValue(7, minute_range)  # 분차트 주기
        self.objStockChart.SetInputValue(9, ord('1'))  # 수정주가 사용

        received_data = {col: [] for col in request_columns}

        while True:
            self.objStockChart.BlockRequest()  # 요청! 후 응답 대기
            self._check_rq_status()  # 통신상태 검사
            time.sleep(0.1)  # 시간당 RQ 제한으로 인해 장애가 발생하지 않도록 딜레이를 줌

            batch_len = self.objStockChart.GetHeaderValue(3)  # 받아온 데이터 개수
            if batch_len == 0:  # 데이터가 없는 경우
                print('데이터 없음')
                return False

            for i in range(batch_len):
                for col_idx, col in enumerate(request_columns):
                    value = self.objStockChart.GetDataValue(col_idx, i)
                    received_data[col].append(value)

            received_oldest_date = min(received_data['date'])
            if received_oldest_date < int(from_date):
                break

        received_df = pd.DataFrame(received_data)
        received_df = received_df[received_df['date'] >= int(from_date)]
        return received_df.sort_values(['date', 'time']).reset_index(drop=True)


    # 차트 요청 - 틱 차트
    @check_PLUS_status
    def request_tick(self, code, tick_range, day_count):
        """
        틱 차트는 최대 20일까지만 제공
        https://money2.creontrade.com/e5/mboard/ptype_basic/Basic_018/DW_Basic_Read_Page.aspx?boardseq=60&seq=16944&page=3&searchString=%ed%8b%b1+%ec%b0%a8%ed%8a%b8&p=8829&v=8637&m=9505
        :param code: 종목코드
        :param tick_range: 1틱 or 5틱, ...
        :param day_count: 요청일수
        """
        assert day_count <= 20, 'tick count must be euqal or less then 20'

        count = 1e6

        self.objStockChart.SetInputValue(0, code)  # 종목코드
        self.objStockChart.SetInputValue(1, ord('2'))  # 개수로 받기
        # self.objStockChart.SetInputValue(2, date)  # 요청종료일
        self.objStockChart.SetInputValue(4, count)  # 조회 개수

        request_list = [(0, 'date'), (1, 'time'), (2, 'price'), (8, 'volume')]  # ohlc 4개 모두 동일 -> price 명칭 사용
        request_numbers = [x[0] for x in request_list]
        request_columns = [x[1] for x in request_list]

        self.objStockChart.SetInputValue(5, request_numbers)
        self.objStockChart.SetInputValue(6, ord('T'))  # 차트 주기 - 틱
        self.objStockChart.SetInputValue(7, tick_range)  # 틱차트 주기
        self.objStockChart.SetInputValue(9, ord('1'))  # 수정주가 사용

        received_data = {col: [] for col in request_columns}

        while True:
            self.objStockChart.BlockRequest()  # 요청! 후 응답 대기
            self._check_rq_status()  # 통신상태 검사
            time.sleep(0.1)  # 시간당 RQ 제한으로 인해 장애가 발생하지 않도록 딜레이를 줌

            batch_len = self.objStockChart.GetHeaderValue(3)  # 받아온 데이터 개수
            if batch_len == 0:  # 20일이 넘어가 데이터가 없는 경우
                break

            initial_date, initial_time = None, None
            for i in range(batch_len):
                for col_idx, col in enumerate(request_columns):
                    value = self.objStockChart.GetDataValue(col_idx, i)
                    received_data[col].append(value)

                    if col == 'date' and not initial_date:
                        initial_date = value
                    if col == 'time' and not initial_time:
                        initial_time = value

            print(f"loaded: {initial_date}_{initial_time} - {received_data['date'][0]}_{received_data['time'][0]}", end='\r')
            received_day_count = pd.Series(received_data['date']).nunique()
            if received_day_count > day_count:
                break

        received_df = pd.DataFrame(received_data)
        oldest_date = min(received_df['date']) + 1
        received_df = received_df[received_df['date'] >= oldest_date]

        print('load complete.')
        return received_df.sort_values(['date', 'time']).reset_index(drop=True)


if __name__ == '__main__':
    objStockChart = CpStockChart()
    # df_daily = objStockChart.request_dwm(code='A000660', dwm='D', from_date='20200101', to_date='20201231')
    df_minutely = objStockChart.request_minute(code='A000660', minute_range=1, from_date='20211004', to_date='20211008')
    # df_tickly = objStockChart.request_tick(code='A000660', tick_range=1, day_count=10)