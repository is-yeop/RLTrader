# 투자 행동을 수행하고 투자금과 보유 주식을 관리하기 위한 클래스
import numpy as np
import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료
    TRADING_TAX = 0.0025  # 거래세

    # 행동
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=0.5):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 단일 매매 단위, 최대 단일 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 보유 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # PV = balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0  # 학습 직전의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profit_loss = 0  # 현재 손익
        self.base_profit_loss = 0  # 직전 지연 보상 직후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준

        # Agent class의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand()/2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price()
        )
        self.ratio_portfolio_value = (
            self.portfolio_value / self.ratio_portfolio_value
        )
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    # 행동 결정 패턴
    def decide_action(self, pred_value, pred_policy, epsilon):
        '''
        :param pred_value:
        :param pred_policy:
        :param epsilon: 무작위 행동을 결정하는 확률, 입력이 없을 시 신경망을 이용함
        :return:
        '''
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            max_pred = np.max(pred)
            if (pred == max_pred).all():
                epsilon = 1

        # 탐험 결정; 들어온 epsilon value에 따라서 분기됨.
        if np.random.rand() < epsilon:
            exploration = True
            # exploration_base가 1에 가까울 수록 매수 신호를 더 많이 보내기 위한 분기
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            # 매도 매수 탐험을 진행함
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1

        # 신경망을 통해서 행동을 결정하게 됨.
        else:
            exploration = False
            action = np.argmx(pred)
            confidence = .5
            if pred_policy is not None:
                confidence = pred[action]
            elif pred_value is not None:
                confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는 지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE
            ) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는 지 확인
            if self.num_stocks <= 0:
                return False

        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit -
                              self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수한다.
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1+self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )

            # 수수료를 적용해 총 매수 금액 선정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 매수 금액이 0이 아닐때 구매 요청
            if invest_amount > 0:
                self.balance -= invest_amount
                self.num_buy += trading_unit
                self.num_buy += 1

        # 매도
        elif action == Agent.ACTION_SELL :
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자를 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도 진행
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)
            ) * trading_unit

            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 주식 보유 수 갱신
                self.balance += invest_amount  # 보유 현금 갱신
                self.num_sell += 1  # 매도 횟수 증가
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profit_loss = (
            (self.portfolio_value - self.initial_balance)
            / self.initial_balance
        )

        # 즉시 보상 - 수익률
        self.immediate_reward = self.profit_loss

        # 지연 보상 - 익절, 손절 기준
        delayed_reward = 0
        self.base_profit_loss = (
            (self.portfolio_value - self.base_portfolio_value)
            / self.base_portfolio_value
        )
        if self.base_profit_loss > self.delayed_reward_threshold or
            self.base_profit_loss < -self.delayed_reward_threshold:
            # 목표 수익률을 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
