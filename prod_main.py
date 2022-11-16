import time
import threading
import numpy as np

from ibapi.client import EClient, TickAttribLast
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from ibapi.common import OrderId
from ibapi.order_state import OrderState
from ibapi.ticktype import TickTypeEnum
from const import data_inf_prod_consts as DC
from lib import prod_helper
from lib.running_lists import RunningListPstBatch, RunningListAvg

AMZN_ID = 1
sym_list = {
    AMZN_ID: 'AMZN',
}

r_l_batch = RunningListPstBatch(seq_len=DC.PAST_LENGTH, batch_size=DC.PROD_BATCH, symbol_id=AMZN_ID)
r_l_avg = RunningListAvg(7, 1)


# @staticmethod
def BracketOrder(
        parentOrderId: int,
        action: str,
        quantity: float,
        limitPrice: float,
        takeProfitLimitPrice: float,
        stopLossPrice: float
):

    limitPrice = np.round(limitPrice, 2)
    takeProfitLimitPrice = np.round(takeProfitLimitPrice, 2)
    stopLossPrice = np.round(stopLossPrice, 2)

    # This will be our main or "parent" order
    parent = Order()
    parent.orderId = parentOrderId
    parent.action = action
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = limitPrice
    # The parent and children orders will need this attribute set to False to prevent accidental executions.
    # The LAST CHILD will have it set to True,
    parent.transmit = False

    takeProfit = Order()
    takeProfit.orderId = parent.orderId + 1
    takeProfit.action = "SELL" if action == "BUY" else "BUY"
    takeProfit.orderType = "LMT"
    takeProfit.totalQuantity = quantity
    takeProfit.lmtPrice = takeProfitLimitPrice
    takeProfit.parentId = parentOrderId
    takeProfit.transmit = False

    stopLoss = Order()
    stopLoss.orderId = parent.orderId + 2
    stopLoss.action = "SELL" if action == "BUY" else "BUY"
    stopLoss.orderType = "STP"
    # Stop trigger price
    stopLoss.auxPrice = stopLossPrice
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parentOrderId
    # In this case, the low side order will be the last child being sent. Therefore, it needs to set this attribute to True
    # to activate all its predecessors
    stopLoss.transmit = True

    bracketOrder = [parent, takeProfit, stopLoss]
    return bracketOrder


def HandleOrder(long_or_short, id_, limit_price, target_price, stop_price):

    if app.ORDER_LOCK_OPEN is True:
        print('closing lock, starting to place orders...')
        app.ORDER_LOCK_OPEN = False

        #  limitPrice, takeProfitLimitPrice, stopLossPrice
        bracket = BracketOrder(app.nextorderId, long_or_short, DC.PROD_NUM_SHARES, limit_price, target_price, stop_price)

        parentID = bracket[0].orderId
        print('Parent ID is ...: ', parentID)

        counter = 0
        for ord in bracket:
            print('Place order called...', ord.orderId, 'counter = ', counter)
            app.placeOrder(ord.orderId, ContractCreator(sym_list[id_]), ord)

            app.order_status_dict[ord.orderId] = 'place order called'
            app.order_child_parent_dict[ord.orderId] = counter  # 0 == parent, 1 and 2 == children

            counter += 1
            app.nextorderId += 1  # need to advance this we'll skip one extra oid, it's fine

        # wait for a few seconds to give chance to fill the order
        time.sleep(7.777)

        if app.order_status_dict[parentID] == 'Filled':
            print('Order', parentID, 'filled...')
        else:
            print('cancel parent and child orders... failed to fill...')
            for ord1 in bracket:
                app.cancelOrder(ord1.orderId)
                print('cancelling order...', ord1.orderId)

            app.ORDER_LOCK_OPEN = True
            print('failed to fill, orders cancelled, loc open')


def CallModel():
    while True:
        if r_l_batch.batch_ready():
            my_batch = r_l_batch.get_batch()

            if app.ORDER_LOCK_OPEN:
                max_signal, min_signal = prod_helper.infer(my_batch)

                if max_signal >= DC.PROD_BUY_THRESHOLD:
                    sym_id = r_l_batch.get_id()
                    running_average = r_l_avg.get_average()

                    limit_price = running_average - DC.PROD_UNDERBID_USD
                    target_price = ((running_average / 100.0) * DC.PROD_TARGET_PERCENT) + running_average
                    stop_price = ((running_average / 100.0) * -DC.PROD_STOP_PERCENT) + running_average

                    order_thread = threading.Thread(target=HandleOrder, args=("BUY", sym_id, limit_price, target_price, stop_price))
                    order_thread.start()

                elif min_signal <= DC.PROD_SELL_THRESHOLD:
                    sym_id = r_l_batch.get_id()
                    running_average = r_l_avg.get_average()

                    limit_price = running_average + DC.PROD_UNDERBID_USD
                    target_price = ((running_average / 100.0) * -DC.PROD_TARGET_PERCENT) + running_average
                    stop_price = ((running_average / 100.0) * DC.PROD_STOP_PERCENT) + running_average

                    order_thread = threading.Thread(target=HandleOrder, args=("SELL", sym_id, limit_price, target_price, stop_price))
                    order_thread.start()


class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.order_child_parent_dict = {}
        self.order_status_dict = {}
        self.ORDER_LOCK_OPEN = True

    def openOrder(self, orderId, contract, order, orderState):
        print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange,
              ':', order.action, order.orderType, order.totalQuantity, orderState.status)

        self.order_status_dict[orderId] = orderState.status

    def position(
            self,
            account: str,
            contract: Contract,
            position: float,
            avgCost: float
    ):
        super().position(account, contract, position, avgCost)
        print("Position..", "Symbol:", contract.symbol, "Position:", position, "Avg cost:", avgCost)

    def positionEnd(self):
        super().positionEnd()
        print("PositionEndddd")

    def error(self, reqId, errorCode: int, errorString: str):
        print('ERRORRRR: ', reqId, errorCode, ':::' + errorString)
        print('if one of the child orders is cancelled, lets see if we can open the lock')

        if reqId in self.order_status_dict.keys():
            if self.order_child_parent_dict[reqId] > 0:
                if self.order_status_dict[reqId] != 'Order Canceled - reason:':

                    self.order_status_dict[reqId] = 'Order Canceled - reason:'
                    print('child order canceled, lock open...')
                    self.ORDER_LOCK_OPEN = True


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def tickByTickAllLast(
            self,
            reqId: int,
            tickType: int,
            time: int,
            price: float,
            size: int,
            tickAtrribLast: TickAttribLast,
            exchange: str,
            specialConditions: str
    ):
        r_l_batch.add_values(price, size)
        r_l_avg.add_value(price)


def ContractCreator(symbol):
    my_contract = Contract()
    my_contract.symbol = symbol
    my_contract.secType = 'STK'
    my_contract.exchange = 'SMART'
    my_contract.currency = 'USD'

    return my_contract


buffer_thread = threading.Thread(target=CallModel)
buffer_thread.start()

# connect to TWS
app = TestApp()
app.nextorderId = None
app.connect('123.456.789.10', 7497, 333)
time.sleep(1)  # Sleep interval to allow time for connection to server

# request data
for sym_key in sym_list:
    print(sym_key, sym_list[sym_key])
    contract_ = ContractCreator(sym_list[sym_key])
    app.reqTickByTickData(sym_key, contract_, "AllLast", 0, False)
    print(contract_)
    time.sleep(1)

app.run()
