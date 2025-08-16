-- create_tables.sql
-- MySQL 8.x / InnoDB / utf8mb4

SET NAMES utf8mb4;
SET time_zone = '+00:00';

-- You can uncomment and set a database if needed
-- CREATE DATABASE IF NOT EXISTS trading DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
-- USE trading;

-- ========================
-- Table: pnl_log
-- ========================
DROP TABLE IF EXISTS pnl_log;
CREATE TABLE pnl_log (
  id            INT NOT NULL AUTO_INCREMENT,
  `timestamp`   BIGINT NOT NULL,         -- epoch ms/us (store as provided)
  pnl           DOUBLE NOT NULL,
  funding_fee   DOUBLE NOT NULL,
  PRIMARY KEY (id),
  KEY idx_pnl_log_ts (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ========================
-- Table: trades_log
-- ========================
DROP TABLE IF EXISTS trades_log;
CREATE TABLE trades_log (
  id               INT NOT NULL AUTO_INCREMENT,
  trade_time       VARCHAR(30) NOT NULL,  -- keep as string per spec
  `state`          VARCHAR(10) NOT NULL,
  average_price    DOUBLE NOT NULL,
  direction        VARCHAR(10) NOT NULL,  -- e.g., 'buy'/'sell'
  instrument_name  VARCHAR(30) NOT NULL,  -- e.g., 'BTC-PERPETUAL'
  contracts        DOUBLE NOT NULL,
  profit_loss      DOUBLE NOT NULL,
  fee              DOUBLE NOT NULL,
  PRIMARY KEY (id),
  KEY idx_trades_log_time (trade_time),
  KEY idx_trades_log_instr (instrument_name),
  KEY idx_trades_log_state (`state`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ========================
-- Table: btc_usdc (OHLCV)
-- ========================
DROP TABLE IF EXISTS btc_usdc;
CREATE TABLE btc_usdc (
  `timestamp`  BIGINT NOT NULL,
  open_price   DOUBLE NOT NULL,
  high_price   DOUBLE NOT NULL,
  low_price    DOUBLE NOT NULL,
  close_price  DOUBLE NOT NULL,
  volume       DOUBLE NOT NULL,
  PRIMARY KEY (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ========================
-- Table: eth_usdc (OHLCV)
-- ========================
DROP TABLE IF EXISTS eth_usdc;
CREATE TABLE eth_usdc (
  `timestamp`  BIGINT NOT NULL,
  open_price   DOUBLE NOT NULL,
  high_price   DOUBLE NOT NULL,
  low_price    DOUBLE NOT NULL,
  close_price  DOUBLE NOT NULL,
  volume       DOUBLE NOT NULL,
  PRIMARY KEY (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ========================
-- Table: sol_usdc (OHLCV)
-- ========================
DROP TABLE IF EXISTS sol_usdc;
CREATE TABLE sol_usdc (
  `timestamp`  BIGINT NOT NULL,
  open_price   DOUBLE NOT NULL,
  high_price   DOUBLE NOT NULL,
  low_price    DOUBLE NOT NULL,
  close_price  DOUBLE NOT NULL,
  volume       DOUBLE NOT NULL,
  PRIMARY KEY (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ========================
-- Table: xrp_usdc (OHLCV)
-- ========================
DROP TABLE IF EXISTS xrp_usdc;
CREATE TABLE xrp_usdc (
  `timestamp`  BIGINT NOT NULL,
  open_price   DOUBLE NOT NULL,
  high_price   DOUBLE NOT NULL,
  low_price    DOUBLE NOT NULL,
  close_price  DOUBLE NOT NULL,
  volume       DOUBLE NOT NULL,
  PRIMARY KEY (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Optional: strict mode safety if you use milliseconds in BIGINT
-- You can declare BIGINT UNSIGNED if your timestamps are always non-negative.
-- Example: change BIGINT to BIGINT UNSIGNED above if desired.