import asyncio
import websockets
import json
import mysql.connector
from datetime import datetime

# **1. Connect to MySQL**
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="btc_options_db"
)
cursor = conn.cursor()

async def truncate():
    try:
        cursor.execute("""
            TRUNCATE TABLE btc_usdc;
        """
                       )
        cursor.execute("""
            TRUNCATE TABLE eth_usdc;
        """
                       )
        cursor.execute("""
            TRUNCATE TABLE sol_usdc;
        """
                       )
        cursor.execute("""
            TRUNCATE TABLE xrp_usdc;
        """
                       )
        conn.commit()
        print("tables are truncated")

    except Exception as e:
        print(f"⚠️ Error processing while truncating: {e}")


# **3. Fetch BTC Price Data from Deribit in Chunks**
async def fetch_btc_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "BTC_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No BTC price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO btc_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ BTC Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


# **4. Fetch DVOL Data from Deribit in Chunks**
async def fetch_eth_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "ETH_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No ETH price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO eth_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ ETH Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


async def fetch_sol_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "SOL_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No SOL price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO sol_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ SOL Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


async def fetch_bnb_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "BNB_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No BNB price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO bnb_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ BNB Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


async def fetch_paxg_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "PAXG_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No PAXG price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO paxg_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ PAXG Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


async def fetch_steth_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "STETH_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No STETH price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO steth_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ STETH Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


async def fetch_xrp_price(num_requests: int):
    try:
        end_timestamp = int(datetime.now().timestamp() * 1000)  # Current time in ms
        for _ in range(num_requests):
            start_timestamp = end_timestamp - (60 * 1000 * 1000)

            msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/get_tradingview_chart_data",
                "params": {
                    "instrument_name": "XRP_USDC",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "resolution": "1"
                }
            }

            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as ws:
                await ws.send(json.dumps(msg))
                response = await ws.recv()
                data = json.loads(response).get("result", {})

                if "ticks" not in data:
                    print(f"⚠️ No XRP price data found for {start_timestamp} - {end_timestamp}")
                    end_timestamp = start_timestamp  # Move the window backward
                    continue

                # Insert into MySQL
                for i in range(len(data["ticks"])):
                    cursor.execute("""
                        INSERT INTO xrp_usdc (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price), high_price=VALUES(high_price), 
                        low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                    """, (
                        data["ticks"][i], data["open"][i], data["high"][i], data["low"][i],
                        data["close"][i], data["volume"][i]
                    ))
                conn.commit()
                print(
                    f"✅ XRP Price Data Saved: {datetime.fromtimestamp(start_timestamp / 1000)} - {datetime.fromtimestamp(end_timestamp / 1000)}")

            end_timestamp = start_timestamp - 1000 * 60  # Move the window backward
    except Exception as e:
        print(f"⚠️ Error processing expiration: {e}")


async def run_price_fetch(tickers: list[str], days: int):
    try:
        await truncate()

        # **2. Define Constants**
        N_PER_REQUEST = 1000  # Max limit per request
        DAYS = days  # How many days in the past
        TOTAL_MINUTES = DAYS * 24 * 60  # Total minutes we want
        NUM_REQUESTS = TOTAL_MINUTES // N_PER_REQUEST

        tasks = []
        if "BTC" in tickers:
            tasks.append(fetch_btc_price(NUM_REQUESTS))
        if "ETH" in tickers:
            tasks.append(fetch_eth_price(NUM_REQUESTS))
        if "SOL" in tickers:
            tasks.append(fetch_sol_price(NUM_REQUESTS))
        if "XRP" in tickers:
            tasks.append(fetch_xrp_price(NUM_REQUESTS))
        if "BNB" in tickers:
            tasks.append(fetch_bnb_price(NUM_REQUESTS))
        if "PAXG" in tickers:
            tasks.append(fetch_paxg_price(NUM_REQUESTS))
        if "STETH" in tickers:
            tasks.append(fetch_steth_price(NUM_REQUESTS))

        if tasks:
            await asyncio.gather(*tasks)
        else:
            print("⚠️ No valid tickers selected.")
    except Exception as e:
        print("❌ Error during gather:", e)
