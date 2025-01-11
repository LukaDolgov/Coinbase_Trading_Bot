#assuming maker operations are taxed at .15 and taker at .25, advanced 2 level on coinbase
#work in percents
#amounts are all in crypto
#bids and asks in crypto order book
#orders are lists made of prices and sizes
#bids are a list of orders
#asks are a list of orders
#volume is in crypto (size of all excecuted orders)

 
class user:
    def __init__(self, USDbalance, CRYPTbalance, orders):
        self.USDbalance = USDbalance
        self.CRYPTbalance = CRYPTbalance
        self.Corderbook = orders

class candle:
    def __init__(self, minute, low, high, open, close, volume):
        self.minute = minute
        self.low = low
        self.high = high
        self.open = open
        self.close = close
        self.volume = volume

#asks are ordered from lowest selling price to highest
#bids are from highest buying price to lowest
#getting price: order_book.pricebook["asks"][0].price, quantity: order_book.pricebook["asks"][0].size
#old versions
                
#amounts in crypto # volume in crypto

#new system
class UserLimitOrder:
    def __init__(self, percentchange, OGPL, totalamount):
        self.percentfilled = float(0)
        self.cancelled = False
        self.terminated = False
        self.totalamount = float(totalamount)
        self.transacted_amount = float(0)  # Common for both bought and sold
        self.percentchange = float(percentchange)
        self.USDval = float(0)
        self.OGPL = float(OGPL)
        self.tax = float(0.15)  # Common tax

    def execute_order(self, CPL, order_book, user, price_key, balance_key, direction):
        # Shared execution logic
        orders = order_book.pricebook[price_key]
        while (
            (float(orders[0].price) >= self.NPL if direction == "sell" else float(orders[0].price) <= self.NPL)
            and self.percentfilled < 100 and self.cancelled == False and self.terminated == False
        ):
            order_size = float(orders[0].size)
            if order_size <= (self.totalamount - self.transacted_amount):
                self.transacted_amount += order_size
                self.USDval += (order_size * float(orders[0].price))
                orders = orders[1:]
                self.percentfilled = (self.transacted_amount / self.totalamount) * 100
            else:
                self.USDval += ((self.totalamount - self.transacted_amount) * float(orders[0].price))
                self.transacted_amount = self.totalamount
                self.percentfilled = 100
                orders = orders[1:]

        # Finalize the order
        if (self.percentfilled >= 100 or self.cancelled) and not self.terminated:
            self.USDval *= (1 - self.tax)
            if balance_key == "USDbalance":
                if getattr(user, balance_key) >= self.USDval:
                    setattr(user, balance_key, getattr(user, balance_key) - self.USDval)
                    setattr(user, "CRYPTbalance", getattr(user, "CRYPTbalance") + self.transacted_amount)
                    self.terminated = True
                    print(f"Executed {direction} order for: {self.USDval} USD")
                else:
                    print(f"Insufficient balance to complete the {direction} order.")
                    self.cancelled = True
            elif balance_key == "CRYPTbalance":
                if getattr(user, balance_key) >= self.totalamount:
                    setattr(user, balance_key, getattr(user, balance_key) - self.transacted_amount)
                    setattr(user, "USDbalance", getattr(user, "USDbalance") + self.USDval)
                    self.terminated = True
                    print(f"Executed {direction} order for: {self.USDval} USD")
                else:
                    print(f"Insufficient balance to complete the {direction} order.")
                    self.cancelled = True

    def from_dict(self, data):
            self.type = data.get("type", self.type)
            self.percentfilled = data.get("percentfilled", self.percentfilled)
            self.cancelled = data.get("cancelled", self.cancelled)
            self.terminated = data.get("terminated", self.terminated)
            self.totalamount = data.get("totalamount", self.totalamount)
            self.transacted_amount = data.get("transacted_amount", self.transacted_amount)
            self.percentchange = data.get("percentchange", self.percentchange)
            self.USDval = data.get("USDval", self.USDval)
            self.OGPL = data.get("OGPL", self.OGPL)
            self.NPL = data.get("NPL", self.NPL)
            self.tax = data.get("tax", self.tax)
            return self
    def to_dict(self):
        return {
            "type": self.type,
            "percentfilled": self.percentfilled,
            "cancelled": self.cancelled,
            "terminated": self.terminated,
            "totalamount": self.totalamount,
            "transacted_amount": self.transacted_amount,
            "percentchange": self.percentchange,
            "USDval": self.USDval,
            "OGPL": self.OGPL,
            "NPL": self.NPL,
            "tax": self.tax,
        }
class UserBuyLimitOrder(UserLimitOrder):
    def __init__(self, percentchange, OGPL, totalamount):
        super().__init__(percentchange, OGPL, totalamount)
        self.type = "limbuy"
        self.NPL = self.OGPL * (1 - (self.percentchange / 100))

    def execute_order(self, CPL, order_book, user):
        super().execute_order(CPL, order_book, user, "asks", "USDbalance", "buy")
class UserSellLimitOrder(UserLimitOrder):
    def __init__(self, percentchange, OGPL, totalamount):
        super().__init__(percentchange, OGPL, totalamount)
        self.type = "limsell"
        self.NPL = self.OGPL * (1 + (self.percentchange / 100))

    def execute_order(self, CPL, order_book, user):
        super().execute_order(CPL, order_book, user, "bids", "CRYPTbalance", "sell")
class UserStopSellLimitOrder(UserLimitOrder):
    def __init__(self, percentchange, OGPL, totalamount):
        super().__init__(percentchange, OGPL, totalamount)
        self.NPL = self.OGPL * (1 - (self.percentchange / 100))
        self.type = "stoplimsell"
        self.status = "inactive"
    def execute_order(self, CPL, order_book, user):
        orders = order_book.pricebook["bids"]
        if float(orders[0].price) <= self.NPL:
            self.status = "active"
        if self.status == "active":
            super().execute_order(CPL, order_book, user, "bids", "CRYPTbalance", "sell")
        
    

"""class bracket_sell_order(): 
    def __init__(self, percentchangepos, percentchangeneg, amount):
        self.type = "bracket"
        self.stoplevel = priceleveldown
        self.percentchangeneg = percentchangeneg
        self.limitorder = sell_limit_order(percentchangepos, amount)
        self.stop = ''
        self.amount = amount
    def set_sell_order(self, currpricelevel):
        if currpricelevel == self.stoplevel:
            self.stop = market_sell_order(self.percentchangeneg, self.priceleveldown, self.amount) """


def check_orders(CPL, order_book, user):
    orders = user.Corderbook
    for i in range(0, len(orders)):
        orders[i].execute_order(CPL, order_book, user)
    return user