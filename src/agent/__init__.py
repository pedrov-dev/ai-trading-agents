"""Decision-layer primitives for the event-driven trading agent."""

from agent.llm_signal import LLMSignalResult, generate_llm_signal
from agent.news_signal import infer_trade_side, select_quote_for_event
from agent.portfolio import PortfolioSnapshot, Position
from agent.risk import RiskCheckResult, RiskConfig, RiskManager, RiskViolation
from agent.signals import Signal, TradeIntent, build_signal
from agent.strategy import SimpleEventDrivenStrategy, StrategyConfig

__all__ = [
    "PortfolioSnapshot",
    "Position",
    "RiskCheckResult",
    "RiskConfig",
    "RiskManager",
    "RiskViolation",
    "Signal",
    "TradeIntent",
    "SimpleEventDrivenStrategy",
    "StrategyConfig",
    "LLMSignalResult",
    "build_signal",
    "generate_llm_signal",
    "infer_trade_side",
    "select_quote_for_event",
]
