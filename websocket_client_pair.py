import asyncio
import sys
import json
import logging
from typing import Dict
from datetime import datetime, timedelta
import websockets
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import secrets  # Secure random nonce generator
import base64
import mysql.connector
import numpy as np
from collections import deque
import sklearn.mixture as mix
from fetch_prices import run_price_fetch
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
from telegram import Update
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

class PairTrading:
    def __init__(self) -> None:
        self.beta_0 = None
        self.beta_1 = None
        self.innovation = deque(maxlen=1000)
        self.innovation_value = None
        self.old_innovation_value = None
        self.old_DDIVF = None
        self.DDIVF = None
        self.price_0 = None
        self.price_1 = None
        self.old_price_0 = None
        self.old_price_1 = None
        self.H_return = None
        self.z_return = None
        self.state = None
        self.old_state = None
        self.regime = None
        self.P = None
        self.R = None
        self.delta = None
        self.Q = None
        self.I = None
        self.j = 1000
        self.k = 1000
        self.hmm= None
        self.selected_subscribe = []


    def linear_OLS(self, x_arr, y_arr):
        x_avg = x_arr.mean()
        y_avg = y_arr.mean()
        s_xy = (x_arr - x_avg) * (y_arr - y_avg).T
        s_x = (x_arr - x_avg) * (x_arr - x_avg).T
        beta_1 = s_xy.sum() / s_x.sum()
        beta_0 = y_avg - beta_1 * x_avg
        return beta_1, beta_0

    # every 1 min data_fetch
    def data_fetch(self):

        table_name_0 = f"{self.selected_subscribe[0].lower()}_usdc"
        # Take data from SQL
        sql_0 =  f"""
            SELECT timestamp, close_price
            FROM {table_name_0}
            ORDER BY timestamp
        """
        cursor.execute(sql_0)
        rows_0 = pd.DataFrame(cursor.fetchall(), columns=["timestamp", table_name_0.split('_')[0]])
        rows_0["timestamp"] = pd.to_datetime(rows_0["timestamp"], unit="ms")  # Change to 'ms' if needed

        table_name_1 = f"{self.selected_subscribe[1].lower()}_usdc"
        # Take data from SQL
        sql_1 =  f"""
            SELECT timestamp, close_price
            FROM {table_name_1}
            ORDER BY timestamp
        """
        cursor.execute(sql_1)
        rows_1 = pd.DataFrame(cursor.fetchall(), columns=["timestamp", table_name_1.split('_')[0]])
        rows_1["timestamp"] = pd.to_datetime(rows_1["timestamp"], unit="ms")  # Change to 'ms' if needed

        # Merge on timestamp
        data = rows_0.merge(rows_1, on="timestamp")
        data.set_index("timestamp", inplace=True)

        self.beta_1, self.beta_0 = self.linear_OLS(np.log(data.iloc[:self.j, 0]), np.log(data.iloc[:self.j, 1]))

        data['Predicted Beta 0'] = self.beta_0
        data['Predicted Beta 1'] = self.beta_1
        data['Innovation'] = 0.0
        data['DDIVF'] = 0.0

        self.P = np.zeros((2, 2))
        self.R = np.array([0.001]).reshape(1, 1)
        self.delta = 0.001
        self.Q = self.delta / (1 - self.delta) * np.diag([1, 1])
        self.I = np.identity(2)
        for t in range(self.j, data.shape[0]):
            z = np.array([np.log(data.iloc[t, 1])]).reshape(1, 1)
            H = np.array([1, np.log(data.iloc[t, 0])]).reshape(1, 2)
            # Prediction
            ## 1. Extrapolate the state
            beta_old = np.array([data['Predicted Beta 0'][t], data['Predicted Beta 1'][t]]).reshape(2,1)
            ## 2. Extrapolate uncertainty
            self.P = self.P + self.Q
            ## 3. Compute innovation
            prediction = np.matmul(H, beta_old)
            innovation = z - prediction
            data['Innovation'][t] = innovation[0][0]
            self.innovation.append(innovation[0][0])
            # Update
            ## 5. Compute the Kalman Gain
            K = np.matmul(self.P, H.T) / (np.matmul(np.matmul(H, self.P), H.T) + self.R)
            ## 6. Update estimate with measurement
            beta_predict = beta_old + np.matmul(K, innovation)
            if t != data.shape[0]-1:
                data['Predicted Beta 0'][t + 1], data['Predicted Beta 1'][t + 1] = beta_predict[0][0], beta_predict[1][0]
                ## 7. Update the estimate uncertainty
                self.P = np.matmul(np.matmul((self.I - np.matmul(K, H)), self.P), (self.I - np.matmul(K, H)).T) + np.matmul(np.matmul(K, self.R), K.T)
            elif t == data.shape[0]-1:
                self.beta_0, self.beta_1 = beta_predict[0][0],  beta_predict[1][0]
                self.P = np.matmul(np.matmul((self.I - np.matmul(K, H)), self.P), (self.I - np.matmul(K, H)).T) + np.matmul(np.matmul(K, self.R), K.T)


        data['H Return'] = np.log1p(data.iloc[:, 0].pct_change())
        data['z Return'] = np.log1p(data.iloc[:, 1].pct_change())
        X = data.iloc[self.j + 5:-1][['Innovation', 'H Return', 'z Return']].values
        self.hmm = mix.GaussianMixture(n_components=3, covariance_type="full", n_init=100, random_state=7).fit(X)
        data['State'] = np.nan
        data['State'][self.j + 5:-1] = self.hmm.predict(X)
        self.price_0 = data.iloc[-1,0]
        self.price_1 = data.iloc[-1,1]
        self.regime = data.groupby("State")["Innovation"].agg(mean_abs_innovation=lambda x: x.abs().mean(),count="count")
        print(self.regime)

    async def KF(self):
        #  #Prediction
        z = np.array([np.log(self.price_1)]).reshape(1, 1)
        H = np.array([1, np.log(self.price_0)]).reshape(1, 2)
        ## 1. Extrapolate the state
        beta_old = np.array([self.beta_0, self.beta_1]).reshape(2, 1)
        ## 2. Extrapolate uncertainty
        self.P = self.P + self.Q
        ## 3. Compute innovation
        prediction = np.matmul(H, beta_old)
        innovation = z - prediction
        self.innovation.append(innovation[0][0])
        self.innovation_value = innovation[0][0]

        if len(self.innovation) == self.j:
            ## 4. Compute ddivf
            l = 200
            w = np.array(self.innovation)  # Convert list to NumPy array
            rho = np.corrcoef(w - w.mean(), np.sign(w - w.mean()))[0, 1]
            V = abs(w - w.mean()) / rho
            S = np.zeros(self.k)
            S[0] = V[:l].mean()
            alpha_values = np.arange(0, 0.5, 0.01)
            min_fess = 999999999.9
            alpha_opt = None
            for alpha in alpha_values:
              for i in range(1, self.k):
                S[i] = alpha * V[i] + (1 - alpha) * S[i-1]
              fess = np.sum(np.square(np.subtract(V[l:], S[l-1:-1])))
              if fess < min_fess:
                min_fess = fess
                alpha_opt = alpha
            for i in range(1, self.k):
              S[i] = alpha_opt * V[i] + (1 - alpha_opt) * S[i-1]
            ddivf = S[-1]
            self.DDIVF = ddivf
        else:
            print(f'# of innovation: {len(self.innovation)}')

        # Update
        ## 5. Compute the Kalman Gain
        K = np.matmul(self.P, H.T) / (np.matmul(np.matmul(H, self.P), H.T) + self.R)
        ## 6. Update estimate with measurement
        beta_predict = beta_old + np.matmul(K, innovation)
        self.beta_0, self.beta_1 = beta_predict[0][0], beta_predict[1][0]
        ## 7. Update the estimate uncertainty
        self.P = np.matmul(np.matmul((self.I - np.matmul(K, H)), self.P), (self.I - np.matmul(K, H)).T) + np.matmul(np.matmul(K, self.R), K.T)
        ## 8. Estimate State using hmm
        x = np.array([[self.innovation_value, self.H_return, self.z_return]])  # shape: (1, 3)
        self.state = self.hmm.predict(x)[0]



class WebSocketClient:
    def __init__(self, ws_connection_url, client_id, private_key) -> None:
        # Instance Variables
        self.ws_connection_url: str = ws_connection_url
        self.client_id: str = client_id
        self.private_key = private_key
        self.timestamp = None
        self.encoded_signature = None
        self.nonce = None
        self.data = None
        self.websocket_client: websockets.WebSocketClientProtocol = None
        self.access_token = None
        self.refresh_token = None
        self.refresh_token_expiry_time = None


    def signature(self) -> None:
        # Generate a timestamp
        self.timestamp = round(datetime.now().timestamp() * 1000)
        # Generate a **secure random nonce**
        self.nonce = secrets.token_hex(16)  # 16-byte hex string
        # Empty data field
        self.data = ""
        # Prepare the data to sign
        data_to_sign = bytes('{}\n{}\n{}'.format(self.timestamp, self.nonce, self.data), "latin-1")
        # Sign the data using the RSA private key with padding and hashing algorithm
        signature = self.private_key.sign(
            data_to_sign,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        self.encoded_signature = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')

    async def establish_heartbeat(self) -> None:
        """
        Requests DBT's `public/set_heartbeat` to
        establish a heartbeat connection.
        """
        msg: Dict = {
                    "jsonrpc": "2.0",
                    "id": 9098,
                    "method": "public/set_heartbeat",
                    "params": {
                              "interval": 10
                               }
                    }

        await self.websocket_client.send(
            json.dumps(
                msg
                )
                )


    async def heartbeat_response(self) -> None:
        """
        Sends the required WebSocket response to
        the Deribit API Heartbeat message.
        """
        msg: Dict = {
                    "jsonrpc": "2.0",
                    "id": 8212,
                    "method": "public/test",
                    "params": {}
                    }

        await self.websocket_client.send(
            json.dumps(
                msg
                )
                )

    async def ws_auth(self):
        """
        Authenticate WebSocket Connection.
        """
        msg = {
            "jsonrpc": "2.0",
            "id": 9929,
            "method": "public/auth",
            "params": {
                "grant_type": "client_signature",
                "client_id": self.client_id,
                "timestamp": self.timestamp,
                "signature": self.encoded_signature,
                "nonce": self.nonce,  # Secure random nonce
                "data": self.data
            }
        }
        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"Request for auth: {msg}")


    async def ws_refresh_auth(self) -> None:

        while True:
            if self.refresh_token_expiry_time is not None:
                if datetime.now() > self.refresh_token_expiry_time:
                    msg: Dict = {
                                "jsonrpc": "2.0",
                                "id": 9929,
                                "method": "public/auth",
                                "params": {
                                          "grant_type": "refresh_token",
                                          "refresh_token": self.refresh_token
                                            }
                                }

                    await self.websocket_client.send(
                        json.dumps(
                            msg
                            )
                            )

            await asyncio.sleep(10)

    async def ws_subscribe(self, operation: str, ws_channel: list) -> None:

        await asyncio.sleep(1)

        msg: Dict = {
                    "jsonrpc": "2.0",
                    "method": f"private/{operation}",
                    "id": 42,
                    "params": {
                        "channels": ws_channel
                        }
                    }

        await self.websocket_client.send(
            json.dumps(
                msg
                )
            )
        logging.info(f"Request for subscribe: {msg}")

        await asyncio.sleep(5)

    async def place_order_buy(self, instrument_name: str, amount: float, price: float = None, time_in_force: str = "fill_or_kill",
                              reduce_only: str = 'false', advanced: str = None , order_type: str = "market", label: str = None, post_only: bool = False):

        msg = {
            "jsonrpc": "2.0",
            "id": 1001,
            "method": "private/buy",
            "params": {
                "instrument_name": instrument_name,
                "amount": amount,
                "type": order_type,
                "price": price,
                "post_only": post_only,
                "reduce_only": reduce_only,
            }
        }
        if label:
            msg["params"]["label"] = label

        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"Buy order sent: {msg}")

    async def place_order_sell(self, instrument_name: str, amount: float, price: float = None, time_in_force: str = "fill_or_kill",
                              reduce_only: str = 'false', advanced: str = None, order_type: str = "market", label: str = None, post_only: bool = False):

        msg = {
            "jsonrpc": "2.0",
            "id": 1001,
            "method": "private/sell",
            "params": {
                "instrument_name": instrument_name,
                "amount": amount,
                "type": order_type,
                "price": price,
                "post_only": post_only,
                "reduce_only": reduce_only,
            }
        }
        if label:
            msg["params"]["label"] = label

        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"Sell order sent: {msg}")

    async def cancel_order(self, order_id: str):

        msg = {
            "jsonrpc": "2.0",
            "id": 1002,
            "method": "private/cancel",
            "params": {
                "order_id": order_id
            }
        }
        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"Cancel order sent: {msg}")

    async def edit_order(self, order_id: str, price: float = None, amount: float = None):

        params = {
            "order_id": order_id
        }
        if price is not None:
            params["price"] = price
        if amount is not None:
            params["amount"] = amount

        msg = {
            "jsonrpc": "2.0",
            "id": 1003,
            "method": "private/edit",
            "params": params
        }
        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"Edit order sent: {msg}")

    async def create_combo(self, order_list: list):

        combo = []
        for leg in order_list:
            combo.append({
                "instrument_name": leg["instrument_name"],
                "amount": leg["amount"],
                "price": leg["price"],
                "direction": leg["direction"]
            })

        msg = {
            "jsonrpc": "2.0",
            "id": 1004,
            "method": "private/create_combo_order",
            "params": {
                "combo_order": combo
            }
        }

        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"Combo order sent: {msg}")

    async def simulate_portfolio(self, simulated_positions: dict, add_positions: str):

        msg = {
            "jsonrpc": "2.0",
            "id": 1005,
            "method": "private/simulate_portfolio",
            "params": {
                "currency": "USDC",
                "add_positions": add_positions,
                "simulated_positions": simulated_positions
            }
        }

        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"🔍 Sent simulate_portfolio request with positions: {simulated_positions}")

    async def get_positions(self, kind: str = "future", currency: str = "USDC"):

        params = {
            "currency": currency
        }
        if kind:
            params["kind"] = kind

        msg = {
            "jsonrpc": "2.0",
            "id": 1006,
            "method": "private/get_positions",
            "params": params
        }
        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"📦 get_positions request sent (kind={kind})")

    async def get_account_summary(self, currency: str = "USDC"):

        params = {"currency": currency}

        msg = {
            "jsonrpc": "2.0",
            "id": 1007,
            "method": "private/get_account_summary",
            "params": params
        }
        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"📦 get_account_summary request sent")

    async def close_position(self, instrument_name: str, type: str= "market", price: float = None):

        msg = {
            "jsonrpc": "2.0",
            "id": 1008,
            "method": "private/close_position",
            "params": {
                "instrument_name": instrument_name,
                "type": type,
                "price": price
            }
        }
        await self.websocket_client.send(json.dumps(msg))
        logging.info(f"close_position order sent: {msg}")


class Execute(PairTrading, WebSocketClient):
    def __init__(self, *args, bot_token, chat_id, **kwargs):
        PairTrading.__init__(self)
        WebSocketClient.__init__(self, *args, **kwargs)
        self.latest_prices: dict = {}
        self.selected_subscribe_channel = []
        self.pre_margin_check = True
        self.portfolio_status = None
        self.portfolio_position = None
        self.pnl = None
        self.funding_fee = None
        self.amount_0 = 0
        self.amount_1 = 0
        self.signal = 0
        self.signal_bool = 0
        self.p_now = None
        self.p_past = None
        self.enabled = None
        self.num_of_request = None
        self.threshold = []
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.existing_amount_map = {}
        self.loop = asyncio.new_event_loop()
        self.start()

    def start(self):
        print("Hi")
        self.get_user_info()
        self.generate_subscribe()
        asyncio.run(run_price_fetch(self.selected_subscribe, self.num_of_request))
        self.signature()
        self.data_fetch()
        self.loop.create_task(self.update_and_execute())
        self.loop.create_task(self.risk_manager())
        self.loop.create_task(self.initialize_telegram_bot())
        self.loop.run_until_complete(self.ws_manager())


    def get_user_info(self):
        user_input = input("Enter Pair (e.g. BTC,ETH): ").strip()
        user_input_position = input("Enter existing amounts (e.g. 1,-2): ").strip()
        user_input_enabled = input("Turn On/Off: ").strip().upper()
        user_input_threshold = input("Enter thresholds (e.g. 28, 45, 32): ").strip()
        user_input_days = input("how many days of price history to use?: ").strip()

        tickers = [ticker.strip().upper() for ticker in user_input.split(",")]
        amounts = [float(a.strip()) for a in user_input_position.split(",")]
        thresholds = [float(a.strip()) for a in user_input_threshold.split(",")]

        if len(tickers) != len(amounts):
            print("❌ Number of tickers and amounts must match.")
            return

        for ticker, amount in zip(tickers, amounts):
            self.selected_subscribe.append(ticker)
            self.existing_amount_map[ticker] = amount

        if user_input_enabled == "ON":
            self.enabled = True
        elif user_input_enabled == "OFF":
            self.enabled = False
        else:
            print("Enter ON or OFF!")
            return

        for threshold in thresholds:
            self.threshold.append(threshold)

        self.num_of_request = int(user_input_days)

        print("✅ Subscribed tickers:", self.selected_subscribe)
        print("📊 Amounts:", self.existing_amount_map)
        print("Strategy:", self.enabled)

    def generate_subscribe(self):
        print(self.selected_subscribe)
        for ticker in self.selected_subscribe:
            self.selected_subscribe_channel.append(f"ticker.{ticker}_USDC-PERPETUAL.100ms")



    async def update_and_execute(self):

        while True:
            await asyncio.sleep(60)
            #Update
            self.old_price_0 = self.price_0
            self.old_price_1 = self.price_1
            self.old_state = self.state
            self.old_innovation_value = self.innovation_value
            self.old_DDIVF = self.DDIVF

            self.price_0 = self.latest_prices[f"{self.selected_subscribe[0]}"]
            self.price_1 = self.latest_prices[f"{self.selected_subscribe[1]}"]

            self.H_return = np.log(self.price_0 / self.old_price_0)
            self.z_return = np.log(self.price_1/ self.old_price_1)

            await self.KF()
            print(f"🔹time = {datetime.now()}")
            print(f"H_return = {self.H_return}")
            print(f"z_return = {self.z_return}")
            print(f"Innovation = {self.innovation_value}")
            print(f"beta_0 = {self.beta_0}")
            print(f"beta_1 = {self.beta_1}")
            print(f"state = {self.state}")
            print(f"DDIVF = {self.DDIVF}")

            # Signal
            reward_map = {state: reward for state, reward in zip(self.regime["mean_abs_innovation"].sort_values().index, self.threshold)}
            self.p_now = reward_map.get(self.state, 45)  # default to 30 if state not found
            self.p_past = reward_map.get(self.old_state, 45)

            if self.old_DDIVF is not None:
                if self.innovation_value < self.p_now * self.DDIVF and self.old_innovation_value > self.p_past * self.old_DDIVF:
                    self.signal = -1
                    self.signal_bool = 1
                elif self.innovation_value > -self.p_now * self.DDIVF and self.old_innovation_value < -self.p_past * self.old_DDIVF:
                    self.signal = 1
                    self.signal_bool = 1
                else:
                    self.signal = 0
                    self.signal_bool = 0


            #Execute
            if self.selected_subscribe[0] == "BTC":
                self.amount_0 = self.signal_bool * (self.signal * 0.04 - self.existing_amount_map[f"{self.selected_subscribe[0]}"])
            elif self.selected_subscribe[0] == "ETH":
                self.amount_0 = self.signal_bool * (self.signal * 1 - self.existing_amount_map[f"{self.selected_subscribe[0]}"])
            elif self.selected_subscribe[0] == "SOL":
                self.amount_0 = self.signal_bool * (self.signal * 25 - self.existing_amount_map[f"{self.selected_subscribe[0]}"])

            if self.selected_subscribe[1] == "ETH":
                self.amount_1 = self.signal_bool * (round(-self.signal * (self.price_0 * self.amount_0/ self.price_1) * self.beta_1, 2) - self.existing_amount_map[f"{self.selected_subscribe[1]}"])
            elif self.selected_subscribe[1] == "SOL":
                self.amount_1 = self.signal_bool * (round(-self.signal * (self.price_0 * self.amount_0/ self.price_1) *  self.beta_1, 1) - self.existing_amount_map[f"{self.selected_subscribe[1]}"])
            elif self.selected_subscribe[1] == "XRP":
                self.amount_1 = self.signal_bool * (int(round((-self.signal * (self.price_0 * self.amount_0 / self.price_1) * self.beta_1) / 10)) * 10 - self.existing_amount_map[f"{self.selected_subscribe[1]}"])

            if (self.enabled == True) and (self.pre_margin_check == True):
                self.trade_time = datetime.now()
                if self.amount_0 < 0:
                    await self.place_order_sell(instrument_name="{0}_USDC-PERPETUAL".format(self.selected_subscribe[0]), amount= abs(self.amount_0), label=f"{self.trade_time}")
                elif self.amount_0 > 0:
                    await self.place_order_buy(instrument_name="{0}_USDC-PERPETUAL".format(self.selected_subscribe[0]), amount=abs(self.amount_0), label=f"{self.trade_time}")

                await asyncio.sleep(1)

                if self.amount_1 > 0:
                    await self.place_order_buy(instrument_name="{0}_USDC-PERPETUAL".format(self.selected_subscribe[1]), amount= abs(self.amount_1), label=f"{self.trade_time}" )
                elif self.amount_1 < 0:
                    await self.place_order_sell(instrument_name="{0}_USDC-PERPETUAL".format(self.selected_subscribe[1]), amount=abs(self.amount_1), label=f"{self.trade_time}")

                self.existing_amount_map[f"{self.selected_subscribe[0]}"] = self.amount_0
                self.existing_amount_map[f"{self.selected_subscribe[1]}"] = self.amount_1


    async def ws_manager(self) -> None:
        async with (websockets.connect(self.ws_connection_url, ping_interval=None, compression=None,
                                       close_timeout=300)
                    as self.websocket_client):
            # Authenticate WebSocket Connection
            print("Hello!")

            await self.ws_auth()

            # Establish Heartbeat
            await self.establish_heartbeat()

            self.loop.create_task(self.ws_refresh_auth())

            await self.ws_subscribe(operation='subscribe', ws_channel=self.selected_subscribe_channel)

            while self.websocket_client.state == websockets.protocol.State.OPEN:
                # while self.websocket_client.open:
                message: bytes = await self.websocket_client.recv()
                message: Dict = json.loads(message)
                # await self.ws_subscribe(operation='subscribe', ws_channel=self.selected_expirations_subscribe)

                if 'id' in list(message):
                    if message['id'] == 9929:
                        if self.refresh_token is None:
                            if message.get("result") is not None:
                                logging.info('Successfully authenticated WebSocket Connection')
                                logging.info(message)
                            else:
                                logging.info('Failed to authenticate WebSocket Connection')
                                logging.info(message)
                                sys.exit(1)
                        else:
                            logging.info('Successfully refreshed the authentication of the WebSocket Connection')
                        self.access_token = message['result']['access_token']
                        self.refresh_token = message['result']['refresh_token']

                        # Refresh Authentication well before the required datetime
                        if message['testnet']:
                            expires_in: int = 300
                        else:
                            expires_in: int = message['result']['expires_in'] - 240

                        self.refresh_token_expiry_time = datetime.now() + timedelta(seconds=expires_in)

                    elif message['id'] == 8212:
                        # Avoid logging Heartbeat messages
                        continue

                    elif message['id'] == 1005:
                        logging.info(f"Result of Simulation: {message}")
                        if message['result'] is not None:
                            if message['result']['margin_balance'] < message['result']['projected_initial_margin'] * 1.2:
                                self.pre_margin_check = False
                            else:
                                self.pre_margin_check = True
                        print(f"pre_margin_check: {self.pre_margin_check}")

                    elif message['id'] == 1006:
                        logging.info(f"Position: {message}")
                        if message['result'] is not None:
                            self.portfolio_position = message["result"]
                            pnl = 0
                            funding_fee = 0
                            for i in range(len(self.portfolio_position)):
                                pnl += self.portfolio_position[i]["total_profit_loss"]
                                funding_fee += self.portfolio_position[i]["realized_funding"]
                            self.pnl = pnl
                            self.funding_fee = funding_fee

                            sql = """INSERT INTO pnl_log (timestamp, pnl, funding_fee) 
                                                                 VALUES (%s, %s, %s)"""
                            values = (message["usIn"], self.pnl, self.funding_fee)
                            cursor.execute(sql, values)
                            conn.commit()

                            print("✅ PnL data inserted at {0}".format(datetime.now()))

                    elif message['id'] == 1001:
                        logging.info(f"Result of executes: {message}")
                        if message['result']["order"]["label"] == f"{self.trade_time}":
                            amount = 0
                            weighted_price = 0
                            profit_loss = 0
                            contracts = 0
                            fee = 0
                            for i in range(len(message["result"]["trades"])):
                                state = message["result"]["trades"][i]["state"]
                                amount += message["result"]["trades"][i]["amount"]
                                weighted_price += message["result"]["trades"][i]["price"] * message["result"]["trades"][i]["amount"]
                                direction = message["result"]["trades"][i]["direction"]
                                instrument_name = message["result"]["trades"][i]["instrument_name"]
                                profit_loss += message["result"]["trades"][i]["profit_loss"]
                                contracts += message["result"]["trades"][i]["contracts"]
                                fee += message["result"]["trades"][i]["fee"]

                            average_price = weighted_price / amount

                            sql = """INSERT INTO trades_log (trade_time, state, average_price, direction, 
                                     instrument_name, contracts, profit_loss, fee) 
                                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
                            values = (self.trade_time, state, average_price, direction, instrument_name, contracts, profit_loss, fee)
                            cursor.execute(sql, values)
                            conn.commit()

                            await self.trade_alarm(f"{state}: {direction} {instrument_name} {contracts} at {self.trade_time}.\n"
                                                   f" Average price:{average_price}\n"
                                                   f" pnl:{profit_loss}")
                            print("✅ Trades data inserted at {0} for {1}".format(datetime.now(), instrument_name))

                    elif message['id'] == 1007:
                        logging.info(f"Account summary: {message}")
                        if message['result'] is not None:
                            self.portfolio_status = message["result"]
                            if self.portfolio_status['margin_balance'] < self.portfolio_status['maintenance_margin'] * 1.1:
                                await self.close_position(instrument_name= "{0}_USDC-PERPETUAL".format(self.selected_subscribe[0]))
                                await self.close_position(instrument_name= "{0}_USDC-PERPETUAL".format(self.selected_subscribe[1]))
                                self.existing_amount_map[f"{self.selected_subscribe[0]}"] = 0
                                self.existing_amount_map[f"{self.selected_subscribe[1]}"] = 0
                                print("Trades are closed.")

                elif 'method' in list(message):
                    # Respond to Heartbeat Message
                    if message['method'] == 'heartbeat':
                        await self.heartbeat_response()

                    elif message['method'] == 'subscription':
                        logging.debug(f"Market Data Received: {message}")
                        channel = message["params"]["channel"]

                        for ticker in self.selected_subscribe:
                            if channel == f"ticker.{ticker}-PERPETUAL.100ms":
                                self.latest_prices[f"{ticker}"] = message["params"]["data"].get("mark_price", None)
                                self.latest_prices["timestamp"] = message["params"]["data"].get('timestamp', None)
                                #print("🔹 Updated {0} Future Price: {1}".format(ticker,self.latest_prices[ticker]))

                            elif channel == f"ticker.{ticker}_USDC-PERPETUAL.100ms":
                                self.latest_prices[f"{ticker}"] = message["params"]["data"].get("mark_price", None)
                                self.latest_prices["timestamp"] = message["params"]["data"].get('timestamp', None)
                                #print("🔹 Updated {0} Future Price: {1}".format(ticker, self.latest_prices[ticker]))
            else:
                logging.info('WebSocket connection has broken.')
                sys.exit(1)


    async def risk_manager(self):
        await asyncio.sleep(100)
        while True:
            await asyncio.sleep(60)

            try:
                amount_0 = 0
                amount_1 = 0

                if self.selected_subscribe[0] == "BTC":
                    amount_0 = 1 * 0.04
                elif self.selected_subscribe[0] == "ETH":
                    amount_0 = 1
                elif self.selected_subscribe[0] == "SOL":
                    amount_0 = 1 * 25

                if self.selected_subscribe[1] == "ETH":
                    amount_1 = float(round(-1 * (self.price_0 * amount_0/ self.price_1) *  self.beta_1, 2))
                elif self.selected_subscribe[1] == "SOL":
                    amount_1 = float(round(-1 * (self.price_0 * amount_0/ self.price_1) *  self.beta_1, 1))
                elif self.selected_subscribe[1] == "XRP":
                    amount_1 = int(round((-1 * (self.price_0 * amount_0 / self.price_1) * self.beta_1) / 10)) * 10
                await self.simulate_portfolio({
                    "{0}_USDC-PERPETUAL".format(self.selected_subscribe[0]): amount_0,
                    "{0}_USDC-PERPETUAL".format(self.selected_subscribe[1]): amount_1
                }, "false")
                await asyncio.sleep(1)

                if self.selected_subscribe[0] == "BTC":
                    amount_0 = -1 * 0.04
                elif self.selected_subscribe[0] == "ETH":
                    amount_0 = -1
                elif self.selected_subscribe[0] == "SOL":
                    amount_0 = -1 * 25

                if self.selected_subscribe[1] == "ETH":
                    amount_1 = float(round(1 * (self.price_0 * amount_0 / self.price_1) * self.beta_1, 2))
                elif self.selected_subscribe[1] == "SOL":
                    amount_1 = float(round(1 * (self.price_0 * amount_0 / self.price_1) * self.beta_1, 1))
                elif self.selected_subscribe[1] == "XRP":
                    amount_1 = int(round((1 * (self.price_0 * amount_0 / self.price_1) * self.beta_1) / 10)) * 10
                await self.simulate_portfolio({
                    "{0}_USDC-PERPETUAL".format(self.selected_subscribe[0]): amount_0,
                    "{0}_USDC-PERPETUAL".format(self.selected_subscribe[1]): amount_1
                }, "false")

                await self.get_positions()
                await self.get_account_summary()

                await asyncio.sleep(5)

            except Exception as e:
                print(f"⚠️ Error processing while simulation: {e}")
                continue

    # **1. Start Command**
    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Welcome to the Trading Bot! 🚀\n"
            "Use the following commands:\n"
            "/trade <buy/sell> <amount> <instrument> [price] - Place an order\n"
            "/cancel <order_id> - Cancel an order\n"
            "/margin - Check margin & P&L\n"
            "/toggle_strategy <on/off> - Enable/Disable Trading\n"
        )

  # **2. Place an Order**
    async def trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if len(args) < 3:
                await update.message.reply_text("Usage: /trade <buy/sell> <amount> <instrument> [price]")
                return

            direction = args[0].lower()
            amount = float(args[1])
            instrument = args[2]
            price = float(args[3]) if len(args) > 3 else None

            if direction not in ["buy", "sell"]:
                await update.message.reply_text("Invalid direction. Use 'buy' or 'sell'.")
                return

            if direction == 'buy':
                await self.place_order_buy(instrument, amount, price)
            elif direction == 'sell':
                await self.place_order_sell(instrument, amount, price)

            await update.message.reply_text("✅ Order placed!")

        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {str(e)}")

    # **3. Cancel an Order**
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if len(args) < 1:
                await update.message.reply_text("Usage: /cancel <order_id>")
                return

            order_id = args[0]
            await self.cancel_order(order_id)
            await update.message.reply_text("❌ Order Canceled")

        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {str(e)}")

    # **4. Check Margin & P&L**
    async def margin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            margin_data = self.portfolio_status
            print(f"margin_data? : {margin_data}")
            await update.message.reply_text(
                f"📊 Portfolio Summary:\n"
                f"🔹 Maintenance_margin: {margin_data['maintenance_margin']} USDC\n"
                f"🔹 Initial margin: {margin_data['initial_margin']} USDC\n"
                f"🔹 Available Margin: {margin_data['margin_balance']} USDC\n"
                f"🔹 P&L: {margin_data['total_pl']} USDC"
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {str(e)}")

    # **5. Toggle Risk Reversal Trading**
    async def toggle_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if len(args) < 1:
                await update.message.reply_text("Usage: /toggle_strategy <on/off>")
                return
            enabled = args[0].lower() == "on"
            self.enabled = enabled
            print(f"enabled? : {self.enabled}")
            await update.message.reply_text(f"⚙️ Strategy is now {'ENABLED' if enabled else 'DISABLED'}.")

        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {str(e)}")

    async def trade_alarm(self, message: str) -> None:
        try:
            await self.bot.send_message(self.chat_id, text=message)
        except Exception as e:
            print(e)

    async def initialize_telegram_bot(self):
        # build the application
        self.bot = telegram.Bot(token=self.bot_token)
        app = ApplicationBuilder().token(self.bot_token).build()
        print("OK?")

        app.add_handler(CommandHandler("start", self.telegram_start))
        app.add_handler(CommandHandler("trade", self.trade))
        app.add_handler(CommandHandler("cancel", self.cancel))
        app.add_handler(CommandHandler("margin", self.margin))
        app.add_handler(CommandHandler("toggle_strategy", self.toggle_strategy))

        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        print("✅ Telegram bot is running...")

if __name__ == "__main__":
    # Logging
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Connect to MySQL**
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="btc_options_db"
    )
    cursor = conn.cursor()

    # Load the client ID and private key from the PEM file
    with open('key/client_id.txt', 'r') as f:
        client_id = f.readline().strip()

    with open('key/private.pem', 'rb') as private_pem:
        private_key = serialization.load_pem_private_key(private_pem.read(), password=None)

    ws_url = "wss://www.deribit.com/ws/api/v2"

    # Telegram bot token
    with open('key/bot_token.txt', 'r') as f:
        bot_token = f.readline().strip()

    with open('key/chat_id.txt', 'r') as f:
        try:
            chat_id = int(f.readline().strip())
        except ValueError:
            raise ValueError("chat_id.txt must contain a valid integer.")

    # Execute
    test = Execute(ws_url, client_id, private_key, bot_token= bot_token, chat_id = chat_id)
