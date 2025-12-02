#!/usr/bin/env python3

"""
Advanced Execution Engine for Optimal Trade Execution

Implements multiple execution algorithms to minimize market impact and improve pricing:
- TWAP (Time-Weighted Average Price)
- Smart Limit Orders with adaptive pricing
- Iceberg Orders for large positions
- Market Impact Analysis
- Liquidity-aware execution
"""

import time
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Available execution algorithms"""
    MARKET = "market"                    # Simple market orders (current)
    SMART_LIMIT = "smart_limit"          # Intelligent limit orders
    TWAP = "twap"                        # Time-Weighted Average Price
    ICEBERG = "iceberg"                  # Break large orders into small pieces
    ADAPTIVE = "adaptive"                # Adaptive algorithm based on conditions
    LIQUIDITY_SEEKING = "liquidity_seeking"  # Hunt for best liquidity


class ExecutionMetrics:
    """Track execution performance metrics"""

    def __init__(self):
        self.total_orders = 0
        self.successful_orders = 0
        self.total_slippage = 0.0
        self.total_volume = 0.0
        self.execution_times = []
        self.price_improvements = []

    def add_execution(self, order_result: Dict):
        """Add execution result to metrics"""
        self.total_orders += 1
        if order_result.get('status') == 'closed':
            self.successful_orders += 1

        if 'slippage' in order_result:
            self.total_slippage += order_result['slippage']

        if 'volume' in order_result:
            self.total_volume += order_result['volume']

        if 'execution_time' in order_result:
            self.execution_times.append(order_result['execution_time'])

        if 'price_improvement' in order_result:
            self.price_improvements.append(order_result['price_improvement'])

    def get_summary(self) -> Dict:
        """Get execution metrics summary"""
        return {
            'success_rate': self.successful_orders / max(1, self.total_orders),
            'avg_slippage': self.total_slippage / max(1, self.successful_orders),
            'avg_execution_time': np.mean(self.execution_times) if self.execution_times else 0,
            'avg_price_improvement': np.mean(self.price_improvements) if self.price_improvements else 0,
            'total_volume': self.total_volume
        }


class AdvancedExecutionEngine:
    """Advanced execution engine with multiple algorithms"""

    def __init__(self, exchange, config: Dict):
        self.exchange = exchange
        self.config = config
        self.metrics = ExecutionMetrics()

        # Execution parameters
        self.default_algorithm = ExecutionAlgorithm(config.get('execution_algorithm', 'smart_limit'))
        self.max_position_slippage = config.get('max_position_slippage', 0.002)  # 20bps max
        self.twap_duration = config.get('twap_duration', 300)  # 5 minutes
        self.iceberg_chunk_size = config.get('iceberg_chunk_size', 0.1)  # 10% chunks
        self.liquidity_threshold = config.get('liquidity_threshold', 10000)  # $10k min liquidity

        # Market microstructure parameters
        self.spread_threshold = config.get('spread_threshold', 0.001)  # 10bps
        self.volume_impact_factor = config.get('volume_impact_factor', 0.0001)
        self.aggressive_threshold = config.get('aggressive_threshold', 0.05)  # 5%

        # Order management
        self.active_orders = {}
        self.execution_history = []

    def execute_trade(self, signal: Dict, algorithm: Optional[ExecutionAlgorithm] = None) -> Dict:
        """Execute trade using specified algorithm"""
        algorithm = algorithm or self.default_algorithm

        # Analyze market conditions
        market_analysis = self._analyze_market_conditions(signal['symbol'])

        # Choose best execution strategy based on conditions
        if algorithm == ExecutionAlgorithm.ADAPTIVE:
            algorithm = self._choose_adaptive_algorithm(signal, market_analysis)

        logger.info(f"🎯 Executing {signal['side']} {signal['symbol']} using {algorithm.value}")

        # Execute based on chosen algorithm
        if algorithm == ExecutionAlgorithm.MARKET:
            return self._execute_market_order(signal)
        elif algorithm == ExecutionAlgorithm.SMART_LIMIT:
            return self._execute_smart_limit(signal, market_analysis)
        elif algorithm == ExecutionAlgorithm.TWAP:
            return self._execute_twap(signal, market_analysis)
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            return self._execute_iceberg(signal, market_analysis)
        elif algorithm == ExecutionAlgorithm.LIQUIDITY_SEEKING:
            return self._execute_liquidity_seeking(signal, market_analysis)
        else:
            logger.warning(f"Unknown algorithm {algorithm}, falling back to smart limit")
            return self._execute_smart_limit(signal, market_analysis)

    def _analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze current market conditions for optimal execution"""
        try:
            # Get orderbook
            orderbook = self.exchange.fetch_order_book(symbol, limit=20)

            # Get recent trades
            trades = self.exchange.fetch_trades(symbol, limit=100)

            # Calculate spread
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0

            # Calculate liquidity
            bid_liquidity = sum([bid[1] for bid in orderbook['bids'][:10]])  # Top 10 levels
            ask_liquidity = sum([ask[1] for ask in orderbook['asks'][:10]])

            # Calculate volatility from recent trades
            if len(trades) >= 2:
                prices = [trade['price'] for trade in trades[-20:]]  # Last 20 trades
                volatility = np.std(prices) / np.mean(prices) if len(prices) > 1 else 0
            else:
                volatility = 0.01  # Default 1%

            # Calculate volume profile
            recent_volume = sum([trade['amount'] for trade in trades[-50:]])  # Last 50 trades
            avg_trade_size = recent_volume / len(trades) if trades else 0

            return {
                'spread': spread,
                'spread_bps': spread * 10000,  # Basis points
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'total_liquidity': bid_liquidity + ask_liquidity,
                'volatility': volatility,
                'recent_volume': recent_volume,
                'avg_trade_size': avg_trade_size,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'orderbook': orderbook,
                'trades': trades[-20:]  # Last 20 trades for reference
            }

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                'spread': 0.001, 'spread_bps': 10, 'bid_liquidity': 0, 'ask_liquidity': 0,
                'total_liquidity': 0, 'volatility': 0.01, 'recent_volume': 0,
                'avg_trade_size': 0, 'best_bid': 0, 'best_ask': 0
            }

    def _choose_adaptive_algorithm(self, signal: Dict, market_analysis: Dict) -> ExecutionAlgorithm:
        """Choose best algorithm based on market conditions"""
        position_size = signal.get('size', 0)
        spread_bps = market_analysis.get('spread_bps', 10)
        liquidity = market_analysis.get('total_liquidity', 0)
        volatility = market_analysis.get('volatility', 0.01)

        # Large position + low liquidity = Iceberg
        if position_size > liquidity * 0.1:  # More than 10% of visible liquidity
            logger.info(f"🧊 Large position vs liquidity, using Iceberg")
            return ExecutionAlgorithm.ICEBERG

        # High volatility + urgent signal = Market
        if volatility > 0.02 and signal.get('urgency', 'normal') == 'high':
            logger.info(f"⚡ High volatility + urgent, using Market")
            return ExecutionAlgorithm.MARKET

        # Wide spread = Liquidity seeking
        if spread_bps > 20:  # More than 20bps spread
            logger.info(f"🔍 Wide spread ({spread_bps:.1f}bps), using Liquidity Seeking")
            return ExecutionAlgorithm.LIQUIDITY_SEEKING

        # Good liquidity + time available = TWAP
        if liquidity > self.liquidity_threshold and signal.get('urgency', 'normal') != 'high':
            logger.info(f"⏱️ Good liquidity + time, using TWAP")
            return ExecutionAlgorithm.TWAP

        # Default to smart limit
        logger.info(f"🎯 Normal conditions, using Smart Limit")
        return ExecutionAlgorithm.SMART_LIMIT

    def _execute_market_order(self, signal: Dict) -> Dict:
        """Execute simple market order (current method)"""
        start_time = datetime.now()

        try:
            symbol = signal['symbol']
            side = signal['side']
            size = signal['size']

            # Get pre-execution price
            ticker = self.exchange.fetch_ticker(symbol)
            pre_price = ticker['last']

            # Place market order
            if side == 'long' or side == 'buy':
                order = self.exchange.create_market_buy_order(symbol, size)
            else:
                order = self.exchange.create_market_sell_order(symbol, size)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate slippage
            executed_price = order.get('average', order.get('price', pre_price))
            slippage = abs(executed_price - pre_price) / pre_price

            result = {
                'order': order,
                'algorithm': 'market',
                'execution_time': execution_time,
                'slippage': slippage,
                'pre_price': pre_price,
                'executed_price': executed_price,
                'status': order.get('status', 'unknown')
            }

            self.metrics.add_execution(result)
            logger.info(f"✅ Market order executed: {executed_price:.2f} (slippage: {slippage*10000:.1f}bps)")

            return result

        except Exception as e:
            logger.error(f"❌ Market order failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_smart_limit(self, signal: Dict, market_analysis: Dict) -> Dict:
        """Execute intelligent limit order with adaptive pricing"""
        start_time = datetime.now()

        try:
            symbol = signal['symbol']
            side = signal['side']
            size = signal['size']
            spread_bps = market_analysis.get('spread_bps', 10)

            # Calculate aggressive limit price
            best_bid = market_analysis.get('best_bid', 0)
            best_ask = market_analysis.get('best_ask', 0)

            if side == 'long' or side == 'buy':
                # For buying, place limit slightly above best bid to get priority
                if spread_bps > 15:  # Wide spread - be more aggressive
                    limit_price = best_bid + (best_ask - best_bid) * 0.3  # 30% into spread
                else:  # Tight spread - just above best bid
                    tick_size = self._get_tick_size(symbol)
                    limit_price = best_bid + tick_size * 2
            else:
                # For selling, place limit slightly below best ask
                if spread_bps > 15:
                    limit_price = best_ask - (best_ask - best_bid) * 0.3  # 30% into spread
                else:
                    tick_size = self._get_tick_size(symbol)
                    limit_price = best_ask - tick_size * 2

            # Place limit order
            try:
                if side == 'long' or side == 'buy':
                    order = self.exchange.create_limit_buy_order(symbol, size, limit_price)
                else:
                    order = self.exchange.create_limit_sell_order(symbol, size, limit_price)
            except Exception as order_error:
                # If limit order fails, fallback to market order
                logger.warning(f"Limit order failed: {order_error}, falling back to market")
                if side == 'long' or side == 'buy':
                    order = self.exchange.create_market_buy_order(symbol, size)
                else:
                    order = self.exchange.create_market_sell_order(symbol, size)

            # Monitor and adjust if needed
            order = self._monitor_limit_order(order, symbol, market_analysis, timeout=30)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate metrics
            if order.get('status') == 'closed':
                executed_price = order.get('average', order.get('price', limit_price))
                reference_price = best_ask if (side == 'long' or side == 'buy') else best_bid
                price_improvement = abs(reference_price - executed_price) / reference_price
            else:
                executed_price = limit_price
                price_improvement = 0

            result = {
                'order': order,
                'algorithm': 'smart_limit',
                'execution_time': execution_time,
                'limit_price': limit_price,
                'executed_price': executed_price,
                'price_improvement': price_improvement,
                'status': order.get('status', 'unknown')
            }

            self.metrics.add_execution(result)
            logger.info(f"✅ Smart limit executed: {executed_price:.2f} (improvement: {price_improvement*10000:.1f}bps)")

            return result

        except Exception as e:
            logger.error(f"❌ Smart limit failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_twap(self, signal: Dict, market_analysis: Dict) -> Dict:
        """Execute Time-Weighted Average Price algorithm"""
        start_time = datetime.now()

        try:
            symbol = signal['symbol']
            side = signal['side']
            total_size = signal['size']
            duration = signal.get('twap_duration', self.twap_duration)  # seconds

            # Calculate number of slices
            num_slices = min(20, max(5, int(duration / 15)))  # 5-20 slices, 15s minimum per slice
            slice_size = total_size / num_slices
            slice_interval = duration / num_slices

            logger.info(f"⏱️ TWAP: {num_slices} slices of {slice_size:.6f} over {duration}s")

            executed_orders = []
            total_executed = 0
            weighted_price = 0

            for i in range(num_slices):
                try:
                    # Adjust size for last slice to handle rounding
                    current_size = total_size - total_executed if i == num_slices - 1 else slice_size

                    # Create slice signal
                    slice_signal = signal.copy()
                    slice_signal['size'] = current_size
                    slice_signal['urgency'] = 'low'  # Don't be aggressive

                    # Execute slice using smart limit
                    slice_result = self._execute_smart_limit(slice_signal, market_analysis)

                    if slice_result.get('status') == 'closed':
                        executed_price = slice_result.get('executed_price', 0)
                        executed_size = slice_result.get('order', {}).get('filled', current_size)

                        executed_orders.append(slice_result)
                        total_executed += executed_size
                        weighted_price += executed_price * executed_size

                        logger.info(f"  Slice {i+1}/{num_slices}: {executed_size:.6f} @ {executed_price:.2f}")

                    # Wait for next slice (except last one)
                    if i < num_slices - 1:
                        time.sleep(slice_interval)

                        # Refresh market analysis
                        market_analysis = self._analyze_market_conditions(symbol)

                except Exception as e:
                    logger.error(f"Error in TWAP slice {i+1}: {e}")
                    continue

            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate TWAP
            if total_executed > 0:
                twap_price = weighted_price / total_executed
                fill_rate = total_executed / total_size
            else:
                twap_price = 0
                fill_rate = 0

            result = {
                'algorithm': 'twap',
                'executed_orders': executed_orders,
                'twap_price': twap_price,
                'total_executed': total_executed,
                'fill_rate': fill_rate,
                'execution_time': execution_time,
                'num_slices': num_slices,
                'status': 'closed' if fill_rate > 0.9 else 'partial'
            }

            self.metrics.add_execution(result)
            logger.info(f"✅ TWAP completed: {twap_price:.2f} ({fill_rate*100:.1f}% filled)")

            return result

        except Exception as e:
            logger.error(f"❌ TWAP failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_iceberg(self, signal: Dict, market_analysis: Dict) -> Dict:
        """Execute iceberg order (large order broken into small visible pieces)"""
        start_time = datetime.now()

        try:
            symbol = signal['symbol']
            side = signal['side']
            total_size = signal['size']

            # Calculate chunk size based on liquidity
            total_liquidity = market_analysis.get('total_liquidity', 0)
            max_chunk = total_liquidity * self.iceberg_chunk_size  # 10% of visible liquidity
            chunk_size = min(total_size * 0.2, max_chunk, total_size)  # Max 20% of order

            logger.info(f"🧊 Iceberg: {total_size:.6f} total, {chunk_size:.6f} chunks")

            executed_orders = []
            remaining_size = total_size
            weighted_price = 0
            total_executed = 0

            while remaining_size > 0:
                try:
                    # Size for this chunk
                    current_chunk = min(chunk_size, remaining_size)

                    # Create chunk signal
                    chunk_signal = signal.copy()
                    chunk_signal['size'] = current_chunk

                    # Execute chunk
                    chunk_result = self._execute_smart_limit(chunk_signal, market_analysis)

                    if chunk_result.get('status') == 'closed':
                        executed_price = chunk_result.get('executed_price', 0)
                        executed_size = chunk_result.get('order', {}).get('filled', current_chunk)

                        executed_orders.append(chunk_result)
                        remaining_size -= executed_size
                        total_executed += executed_size
                        weighted_price += executed_price * executed_size

                        logger.info(f"  Chunk: {executed_size:.6f} @ {executed_price:.2f} (remaining: {remaining_size:.6f})")
                    else:
                        # If chunk failed, wait and retry with smaller size
                        chunk_size *= 0.8  # Reduce chunk size
                        time.sleep(5)
                        continue

                    # Wait between chunks to avoid detection
                    if remaining_size > 0:
                        wait_time = np.random.uniform(10, 30)  # 10-30 seconds
                        time.sleep(wait_time)

                        # Refresh market conditions
                        market_analysis = self._analyze_market_conditions(symbol)

                except Exception as e:
                    logger.error(f"Error in iceberg chunk: {e}")
                    break

            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate metrics
            if total_executed > 0:
                avg_price = weighted_price / total_executed
                fill_rate = total_executed / total_size
            else:
                avg_price = 0
                fill_rate = 0

            result = {
                'algorithm': 'iceberg',
                'executed_orders': executed_orders,
                'avg_price': avg_price,
                'total_executed': total_executed,
                'fill_rate': fill_rate,
                'execution_time': execution_time,
                'num_chunks': len(executed_orders),
                'status': 'closed' if fill_rate > 0.9 else 'partial'
            }

            self.metrics.add_execution(result)
            logger.info(f"✅ Iceberg completed: {avg_price:.2f} ({fill_rate*100:.1f}% filled)")

            return result

        except Exception as e:
            logger.error(f"❌ Iceberg failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_liquidity_seeking(self, signal: Dict, market_analysis: Dict) -> Dict:
        """Execute liquidity-seeking algorithm (hunt for best liquidity)"""
        start_time = datetime.now()

        try:
            symbol = signal['symbol']
            side = signal['side']
            size = signal['size']

            # Start conservatively in the spread
            best_bid = market_analysis.get('best_bid', 0)
            best_ask = market_analysis.get('best_ask', 0)

            if side == 'long' or side == 'buy':
                # Start at midpoint, slowly move towards ask
                start_price = (best_bid + best_ask) / 2
                target_price = best_ask
                price_step = (target_price - start_price) / 10
            else:
                # Start at midpoint, slowly move towards bid
                start_price = (best_bid + best_ask) / 2
                target_price = best_bid
                price_step = (start_price - target_price) / 10

            logger.info(f"🔍 Liquidity seeking: {start_price:.2f} -> {target_price:.2f}")

            current_price = start_price
            attempts = 0
            max_attempts = 10

            while attempts < max_attempts:
                try:
                    # Place limit order at current price level
                    if side == 'long' or side == 'buy':
                        order = self.exchange.create_limit_buy_order(symbol, size, current_price)
                    else:
                        order = self.exchange.create_limit_sell_order(symbol, size, current_price)

                    # Wait for partial or full fill
                    order = self._monitor_limit_order(order, symbol, market_analysis, timeout=20)

                    if order.get('status') == 'closed':
                        # Fully filled
                        break
                    elif order.get('filled', 0) > 0:
                        # Partially filled - adjust size and continue
                        filled = order.get('filled', 0)
                        size -= filled
                        logger.info(f"  Partial fill: {filled:.6f} @ {current_price:.2f}")

                        if size <= 0:
                            break
                    else:
                        # No fill - cancel and move price closer to market
                        try:
                            self.exchange.cancel_order(order['id'], symbol)
                        except:
                            pass

                    # Move price closer to market
                    if side == 'long' or side == 'buy':
                        current_price += price_step
                    else:
                        current_price -= price_step

                    attempts += 1

                except Exception as e:
                    logger.error(f"Error in liquidity seeking attempt {attempts}: {e}")
                    attempts += 1
                    continue

            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate final metrics
            executed_price = order.get('average', order.get('price', current_price)) if 'order' in locals() else current_price
            fill_rate = 1.0 - (size / signal['size']) if size < signal['size'] else 0

            result = {
                'order': order if 'order' in locals() else None,
                'algorithm': 'liquidity_seeking',
                'execution_time': execution_time,
                'executed_price': executed_price,
                'fill_rate': fill_rate,
                'attempts': attempts,
                'status': 'closed' if fill_rate > 0.9 else 'partial'
            }

            self.metrics.add_execution(result)
            logger.info(f"✅ Liquidity seeking completed: {executed_price:.2f} ({fill_rate*100:.1f}% filled)")

            return result

        except Exception as e:
            logger.error(f"❌ Liquidity seeking failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _monitor_limit_order(self, order: Dict, symbol: str, market_analysis: Dict, timeout: int = 30) -> Dict:
        """Monitor limit order and adjust if needed"""
        order_id = order.get('id')
        if not order_id:
            return order

        start_time = time.time()
        check_interval = 5  # Check every 5 seconds

        while time.time() - start_time < timeout:
            try:
                # Check order status
                updated_order = self.exchange.fetch_order(order_id, symbol)

                if updated_order.get('status') in ['closed', 'canceled']:
                    return updated_order

                # Check if we should adjust price (halfway through timeout)
                if time.time() - start_time > timeout / 2:
                    # Get fresh market data
                    current_analysis = self._analyze_market_conditions(symbol)

                    # If spread changed significantly, consider adjusting
                    original_spread = market_analysis.get('spread_bps', 10)
                    current_spread = current_analysis.get('spread_bps', 10)

                    if abs(current_spread - original_spread) > 5:  # 5bps change
                        logger.info(f"  Spread changed from {original_spread:.1f} to {current_spread:.1f}bps")
                        # Could implement price adjustment logic here

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                break

        # Timeout reached - return last known status
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except:
            return order

    def _get_tick_size(self, symbol: str) -> float:
        """Get minimum price increment for symbol"""
        try:
            market = self.exchange.market(symbol)
            return market.get('precision', {}).get('price', 0.01)
        except:
            # Default tick size based on price level
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get('last', 100)

            if price > 100000:
                return 1.0
            elif price > 10000:
                return 0.1
            elif price > 1000:
                return 0.01
            else:
                return 0.001

    def get_execution_summary(self) -> Dict:
        """Get execution performance summary"""
        return {
            'metrics': self.metrics.get_summary(),
            'active_orders': len(self.active_orders),
            'algorithm': self.default_algorithm.value,
            'config': {
                'max_slippage': self.max_position_slippage,
                'twap_duration': self.twap_duration,
                'iceberg_chunk_size': self.iceberg_chunk_size,
                'liquidity_threshold': self.liquidity_threshold
            }
        }