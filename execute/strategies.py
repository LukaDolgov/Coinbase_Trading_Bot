from classes import user, candle, check_orders, UserLimitOrder, UserBuyLimitOrder, UserSellLimitOrder, UserStopSellLimitOrder
#ENT = entr
#IPSH = in progress sell-high
#IPSL = in progress sell-low
#RR is Reward:Risk
class trade():
    def __init__(self):
        self.Entry = []
        self.IPSHtrade = []
        self.IPSLtrade = []
        self.won = ""

#Micheal Harris Trading Pattern https://www.youtube.com/watch?v=H23GLHD__yY&list=PLwEOixRFAUxZmM26EYI1uYtJG39HDW1zm #mabye longer term more effective?
class strategy1():
    def __init__(self):
        self.candles = []
        self.RR = 6/4
        self.numtrades = 0
        self.currmin = 0
        self.winrate = 0
        self.reset = 0
        self.ENTtrades = []
        self.IPSHtrades = []
        self.IPSLtrades = []
    def execute(self, CPL, order_book, user):
        orders = user.Corderbook
        candlesR = self.candles
        currminR = self.currmin - 1
        self.winrate = self.calculate_winrate()
        #conditions
        print(len(candlesR))
        #quick reset
        if len(self.ENTtrades) > 0 and len(candlesR) - self.reset > 10:
            self.ENTtrades = []
        if len(candlesR) - self.reset > 4:
            c1 = candlesR[currminR].high > candlesR[currminR - 1].high
            c2 = candlesR[currminR - 1].high > candlesR[currminR].low
            c3 = candlesR[currminR].high > candlesR[currminR - 2].high
            c4 = candlesR[currminR - 2].high > candlesR[currminR - 1].low 
            c5 = candlesR[currminR - 1].low > candlesR[currminR - 3].high
            c6 = candlesR[currminR - 3].high > candlesR[currminR - 2].low
            c7 = candlesR[currminR - 2].low > candlesR[currminR - 3].low
            if c1 and c2 and c3 and c4 and c5 and c6 and c7 and len(self.ENTtrades) == 0:
                trade = UserBuyLimitOrder(0.01, float(CPL), 0.05) #0.05 bitcoin 
                orders.append(trade) 
                self.ENTtrades.append(trade)
                self.reset = len(candlesR)
            if len(self.ENTtrades) > 0:
                if self.ENTtrades[0].terminated == True:
                    self.ENTtrades = []
                    tradeup = UserSellLimitOrder(0.6, float(CPL * 0.9999), 0.05) # .6 reward to .4 loss
                    tradedown = UserStopSellLimitOrder(0.4, float(CPL * 0.9999), 0.05)
                    orders.append(tradeup)
                    orders.append(tradedown)
                    self.IPSHtrades.append(tradeup)
                    self.IPSLtrades.append(tradedown)
        for i in range(len(self.IPSHtrades)):
            if self.IPSHtrades[i].terminated:
                orders.remove(self.IPSLtrades[i])
                self.numtrades += 1
            if self.IPSLtrades[i].terminated:
                orders.remove(self.IPSHtrades[i])
                self.numtrades += 1
        return user
    def calculate_winrate(self):
        countwins = 0
        countlosses = 1
        for trade in self.IPSHtrades:
            if trade.terminated == True:
                countwins += 1
        for trade in self.IPSLtrades:
            if trade.terminated == True:
                countlosses += 1
        return float((countwins + 1) / countlosses) * 100 #in percent

#LSTM model Long-term Trading and analysis
#Kelly Criterion Formula for determining staking equity 

def KC_formula(prob_success, to_win):
    fraction_to_invest = (float(prob_success * to_win) - (1 - prob_success)) / float(to_win)
    return fraction_to_invest
    
class strategy2():
    def __init__(self):
        self.candles = []
        self.RR = 6/3
        self.numtrades = 0
        self.currmin = 0
        self.winrate = 0
        self.reset = 0
        self.ENTtrades = []
        self.IPSHtrades = []
        self.IPSLtrades = []
        self.price_prediction = 0
        self.single_exec = False
    def execute(self, CPL, order_book, user):
        orders = user.Corderbook
        candlesR = self.candles
        currminR = self.currmin - 1
        confidence = 0
        percent_compare = self.price_prediction / CPL
        if percent_compare > 1 and self.single_exec == False:
            real_percent_compare = percent_compare - 1
            if real_percent_compare > 0.05:
                confidence = .75
            elif real_percent_compare < 0.025:
                confidence = .25
            else:
                confidence = .5
            b = self.RR
            optimal_stake = KC_formula(confidence, b)
            possible_cryptobuy = float(user.USDbalance / CPL)
            amount_to_buy = possible_cryptobuy * optimal_stake
            trade = UserBuyLimitOrder(0.01, float(CPL), amount_to_buy) #dynamically adjusted buy of bitcoin 
            orders.append(trade) 
            self.ENTtrades.append(trade)
            self.reset = len(candlesR)
        if len(self.ENTtrades) > 0:
            if self.ENTtrades[0].terminated == True:
                self.ENTtrades = []
                tradeup = UserSellLimitOrder(6, float(CPL * 0.9999), amount_to_buy) # 6% reward to 4% loss
                tradedown = UserStopSellLimitOrder(3, float(CPL * 0.9999), amount_to_buy)
                orders.append(tradeup)
                orders.append(tradedown)
                self.IPSHtrades.append(tradeup)
                self.IPSLtrades.append(tradedown)
        for i in range(len(self.IPSHtrades)):
            if self.IPSHtrades[i].terminated:
                orders.remove(self.IPSLtrades[i])
                self.numtrades += 1
            if self.IPSLtrades[i].terminated:
                orders.remove(self.IPSHtrades[i])
                self.numtrades += 1
        self.single_exec = True
        return user
            
            
            
        pass
    