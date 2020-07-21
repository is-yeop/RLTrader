# 투자할 종목의 차트 데이터를 관리함, 과거부터 순차적으로 순회함


class Environment:
    """
    :var chart_data: 주식 종목의 차트 데이터
    :var observation: 현재 관측치
    :var idx: 차트 데이터에서의 현재 위치
    """
    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None):
        """
        :param chart_data: 관리할 차트 데이터를 할당, 2차원 배열
        """
        self.char_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self):
        """
        idx와 observation을 초기화
        """
        self.observation = None
        self.idx = -1

    def observe(self):
        """
        :return: 하루 앞으로 이동하며 차트 데이터에서 관측 데이터를 제공
                 만약 제공할게 없다면 None을 반환한다.
        """
        if len(self.char_data) > self.idx + 1 :
            self.idx += 1
            # iloc은 dataframe 내장 함수로, 특정 행의 데이터를 가져온다.
            self.observation = self.char_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        """
        :return: 관측 데이터로 부터 종가를 가져와서 반환함
                 없으면 None을 반환
        """
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        """
        :param chart_data:
        :return:
        """
        self.chart_data=chart_data
