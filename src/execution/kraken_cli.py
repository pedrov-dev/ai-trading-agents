"""Kraken CLI-style order translation for exchange-backed paper or live trading."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from agent.signals import TradeIntent
from execution.orders import (
    ExecutionMode,
    ExecutionResult,
    OrderAttempt,
    OrderFailure,
    OrderFill,
    OrderRequest,
    OrderStatus,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_LOG_PATH = ROOT_DIR / "artifacts" / "orders_audit.jsonl"

_SYMBOL_TO_KRAKEN_PAIR: dict[str, str] = {
    "btc_usd": "XBT/USD",
    "eth_usd": "ETH/USD",
    "sol_usd": "SOL/USD",
    "xrp_usd": "XRP/USD",
}
_DEFAULT_KRAKEN_API_URL = "https://api.kraken.com"


def _default_cli_executable() -> str:
    script_name = "kraken-cli.exe" if os.name == "nt" else "kraken-cli"
    executable_path = Path(sys.executable)
    candidates: list[Path] = [executable_path.with_name(script_name)]

    try:
        candidates.append(executable_path.resolve().with_name(script_name))
    except OSError:
        pass

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)
    return "kraken-cli"


def _normalize_cli_executable(executable: str | None) -> str:
    default_executable = _default_cli_executable()
    configured = (executable or "").strip()
    if not configured:
        return default_executable

    if configured.lower() in {"kraken-cli", "kraken-cli.exe"}:
        default_path = Path(default_executable)
        if default_path.exists():
            return str(default_path)
    return configured


@dataclass(frozen=True)
class CommandRunResult:
    """Outcome of running one Kraken CLI command."""

    exit_code: int
    stdout: str
    stderr: str


CommandRunner = Callable[[tuple[str, ...], int], CommandRunResult]


@dataclass(frozen=True)
class KrakenCLIConfig:
    """Configuration for the Kraken CLI adapter and audit logging."""

    executable: str = field(default_factory=_default_cli_executable)
    subcommand: tuple[str, ...] = ("add-order",)
    dry_run: bool = True
    live_enabled: bool = False
    validate_only: bool = True
    default_order_type: str = "market"
    timeout_seconds: int = 15
    max_retries: int = 2
    audit_log_path: Path = field(default_factory=lambda: DEFAULT_AUDIT_LOG_PATH)
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> KrakenCLIConfig:
        """Create a config from environment variables using safe defaults."""
        source = env or os.environ
        return cls(
            executable=_normalize_cli_executable(
                source.get("KRAKEN_CLI_EXECUTABLE")
            ),
            dry_run=_parse_bool(source.get("KRAKEN_EXECUTION_DRY_RUN"), default=True),
            live_enabled=_parse_bool(source.get("KRAKEN_LIVE_ENABLED"), default=False),
            validate_only=_parse_bool(source.get("KRAKEN_VALIDATE_ONLY"), default=True),
            default_order_type=source.get("KRAKEN_ORDER_TYPE", "market"),
            timeout_seconds=int(source.get("KRAKEN_CLI_TIMEOUT_SECONDS", "15")),
            max_retries=int(source.get("KRAKEN_CLI_MAX_RETRIES", "2")),
            audit_log_path=Path(
                source.get("KRAKEN_AUDIT_LOG_PATH", str(DEFAULT_AUDIT_LOG_PATH))
            ),
        )


class KrakenCLIExecutor:
    """Translate trade intents into auditable Kraken CLI-style order submissions."""

    def __init__(
        self,
        *,
        config: KrakenCLIConfig | None = None,
        runner: CommandRunner | None = None,
        now_provider: Callable[[], datetime] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config or KrakenCLIConfig.from_env()
        resolved_env = {key: str(value) for key, value in os.environ.items()}
        if env is not None:
            resolved_env.update({key: str(value) for key, value in env.items()})
        self._env = resolved_env
        self._runner = runner or (
            lambda command, timeout_seconds: _run_command(
                command,
                timeout_seconds,
                env=self._env,
            )
        )
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def build_command(self, intent_or_request: TradeIntent | OrderRequest) -> tuple[str, ...]:
        """Translate one trade intent or order request into Kraken CLI arguments."""
        request = self._coerce_request(intent_or_request)
        command = [
            self._config.executable,
            *self._config.subcommand,
            "--pair",
            _normalize_pair(request.symbol_id),
            "--side",
            request.side,
            "--type",
            request.order_type,
            "--volume",
            _format_quantity(request.quantity),
            "--clordid",
            request.client_order_id,
            *self._config.extra_args,
        ]

        if (
            request.execution_mode == ExecutionMode.DRY_RUN
            or self._config.dry_run
            or self._config.validate_only
        ):
            command.append("--validate")

        return tuple(command)

    def submit_trade_intent(self, intent: TradeIntent) -> ExecutionResult:
        """Create an order request from a trade intent and submit it safely."""
        return self.submit_order(self._coerce_request(intent))

    def submit_order(self, request: OrderRequest) -> ExecutionResult:
        """Submit one order request to Kraken, using validation for paper mode."""
        command = self.build_command(request)
        self._write_audit(
            "order_requested",
            request=request,
            status=OrderStatus.REQUESTED,
            extra={"command": list(command)},
        )

        if request.execution_mode == ExecutionMode.DRY_RUN or not self._config.live_enabled:
            return self._simulate_order(request=request, command=command)

        attempts: list[OrderAttempt] = []
        for attempt_number in range(1, self._config.max_retries + 2):
            started_at = self._now_provider()
            run_result = self._runner(command, self._config.timeout_seconds)
            finished_at = self._now_provider()
            response = _parse_json_object(run_result.stdout)

            if run_result.exit_code == 0:
                validation_only = "--validate" in command
                status = OrderStatus.VALIDATED if validation_only else OrderStatus.SUBMITTED
                if not validation_only and response.get("status") == "filled":
                    status = OrderStatus.FILLED
                attempt = OrderAttempt(
                    attempt_number=attempt_number,
                    command=command,
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    exit_code=run_result.exit_code,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                    response=response or None,
                    retryable=False,
                )
                attempts.append(attempt)
                fill = None
                event_name = "order_validated" if validation_only else "order_submitted"
                if validation_only:
                    fill = OrderFill(
                        status=OrderStatus.SIMULATED,
                        filled_quantity=request.quantity,
                        average_price=request.current_price,
                        filled_at=finished_at,
                    )
                if status == OrderStatus.FILLED:
                    fill = OrderFill(
                        status=OrderStatus.FILLED,
                        filled_quantity=float(response.get("filled_quantity", request.quantity)),
                        average_price=float(response.get("average_price", request.current_price)),
                        filled_at=finished_at,
                    )
                    event_name = "order_filled"
                self._write_audit(
                    event_name,
                    request=request,
                    attempt=attempt,
                    fill=fill,
                    status=status,
                )
                return ExecutionResult(
                    request=request,
                    status=status,
                    attempts=tuple(attempts),
                    fill=fill,
                    completed_at=finished_at,
                )

            retryable = _is_retryable_failure(run_result.exit_code, run_result.stderr)
            should_retry = retryable and attempt_number <= self._config.max_retries
            attempt_status = OrderStatus.RETRYING if should_retry else OrderStatus.FAILED
            attempt = OrderAttempt(
                attempt_number=attempt_number,
                command=command,
                status=attempt_status,
                started_at=started_at,
                finished_at=finished_at,
                exit_code=run_result.exit_code,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                response=response or None,
                retryable=retryable,
            )
            attempts.append(attempt)
            self._write_audit(
                "order_retry" if should_retry else "order_failed",
                request=request,
                attempt=attempt,
                status=attempt_status,
            )
            if should_retry:
                continue

            failure_message = (
                run_result.stderr or run_result.stdout or "Kraken CLI execution failed."
            )
            failure = OrderFailure(
                code=f"cli_exit_{run_result.exit_code}",
                message=failure_message,
                occurred_at=finished_at,
                retryable=retryable,
            )
            return ExecutionResult(
                request=request,
                status=OrderStatus.FAILED,
                attempts=tuple(attempts),
                failure=failure,
                completed_at=finished_at,
            )

        fallback_time = self._now_provider()
        failure = OrderFailure(
            code="unexpected_state",
            message="Order processing exited unexpectedly without a terminal result.",
            occurred_at=fallback_time,
            retryable=False,
        )
        return ExecutionResult(
            request=request,
            status=OrderStatus.FAILED,
            attempts=tuple(attempts),
            failure=failure,
            completed_at=fallback_time,
        )

    def _coerce_request(self, intent_or_request: TradeIntent | OrderRequest) -> OrderRequest:
        if isinstance(intent_or_request, OrderRequest):
            return intent_or_request
        execution_mode = ExecutionMode.DRY_RUN if self._config.dry_run else ExecutionMode.LIVE
        return OrderRequest.from_trade_intent(
            intent_or_request,
            execution_mode=execution_mode,
            order_type=self._config.default_order_type,
        )

    def _simulate_order(
        self,
        *,
        request: OrderRequest,
        command: tuple[str, ...],
    ) -> ExecutionResult:
        simulated_at = self._now_provider()
        attempt = OrderAttempt(
            attempt_number=1,
            command=command,
            status=OrderStatus.SIMULATED,
            started_at=simulated_at,
            finished_at=simulated_at,
            exit_code=0,
            stdout="dry_run: command not executed",
            stderr="",
            response={"simulated": True, "mode": request.execution_mode.value},
            retryable=False,
        )
        fill = OrderFill(
            status=OrderStatus.SIMULATED,
            filled_quantity=request.quantity,
            average_price=request.current_price,
            filled_at=simulated_at,
        )
        result = ExecutionResult(
            request=request,
            status=OrderStatus.SIMULATED,
            attempts=(attempt,),
            fill=fill,
            completed_at=simulated_at,
        )
        self._write_audit(
            "order_simulated",
            request=request,
            attempt=attempt,
            fill=fill,
            status=OrderStatus.SIMULATED,
        )
        return result

    def _write_audit(
        self,
        event: str,
        *,
        request: OrderRequest,
        status: OrderStatus,
        attempt: OrderAttempt | None = None,
        fill: OrderFill | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "event": event,
            "recorded_at": self._now_provider().isoformat(),
            "status": status.value,
            "request": request.to_dict(),
        }
        if attempt is not None:
            entry["attempt"] = attempt.to_dict()
        if fill is not None:
            entry["fill"] = fill.to_dict()
        if extra:
            entry.update(extra)

        self._config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._config.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _run_command(
    command: tuple[str, ...],
    timeout_seconds: int,
    *,
    env: Mapping[str, str] | None = None,
) -> CommandRunResult:
    """Execute one CLI command and capture its output safely."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=dict(env) if env is not None else None,
        )
    except FileNotFoundError as exc:
        return CommandRunResult(exit_code=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandRunResult(
            exit_code=124,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr) or str(exc),
        )

    return CommandRunResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_pair(symbol_id: str) -> str:
    if symbol_id in _SYMBOL_TO_KRAKEN_PAIR:
        return _SYMBOL_TO_KRAKEN_PAIR[symbol_id]
    return symbol_id.replace("_", "/").upper()


def _format_quantity(quantity: float) -> str:
    formatted = f"{quantity:.8f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    if not raw_text.strip():
        return {}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _is_retryable_failure(exit_code: int, stderr: str) -> bool:
    retryable_markers = (
        "timeout",
        "temporary",
        "temporarily",
        "network",
        "rate limit",
        "unavailable",
        "try again",
    )
    lowered = stderr.lower()
    return exit_code in {75, 124} or any(marker in lowered for marker in retryable_markers)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-cli",
        description="Local Kraken CLI shim used by the trading agent execution layer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_order = subparsers.add_parser(
        "add-order",
        help="Validate or submit one Kraken spot order.",
    )
    add_order.add_argument(
        "--pair",
        required=True,
        help="Kraken pair like XBT/USD or XBTUSD",
    )
    add_order.add_argument("--side", required=True, choices=("buy", "sell"))
    add_order.add_argument(
        "--type",
        dest="order_type",
        required=True,
        choices=("market", "limit"),
        help="Order type to send to Kraken.",
    )
    add_order.add_argument("--volume", required=True, help="Base asset volume")
    add_order.add_argument("--clordid", required=True, help="Client order id for auditing")
    add_order.add_argument("--price", help="Required for limit orders")
    add_order.add_argument(
        "--validate",
        action="store_true",
        help="Validate the order without live submission.",
    )
    return parser


def _validate_cli_inputs(*, pair: str, side: str, order_type: str, volume: str) -> float:
    normalized_pair = pair.strip()
    if not normalized_pair or ("/" not in normalized_pair and len(normalized_pair) < 6):
        raise ValueError("`--pair` must look like XBT/USD or XBTUSD.")
    if side not in {"buy", "sell"}:
        raise ValueError("`--side` must be `buy` or `sell`.")
    if order_type not in {"market", "limit"}:
        raise ValueError("`--type` must be `market` or `limit`.")

    parsed_volume = float(volume)
    if parsed_volume <= 0:
        raise ValueError("`--volume` must be positive.")
    return parsed_volume


def _live_submit_enabled(env: Mapping[str, str]) -> bool:
    return _parse_bool(env.get("KRAKEN_CLI_ALLOW_LIVE_SUBMIT"), default=False)


def _has_private_api_credentials(env: Mapping[str, str]) -> bool:
    return bool(env.get("KRAKEN_API_KEY", "").strip()) and bool(
        env.get("KRAKEN_API_SECRET", "").strip()
    )


def _normalize_private_api_pair(pair: str) -> str:
    normalized = pair.strip().upper()
    aliases = {
        "BTC/USD": "XBTUSD",
        "BTCUSD": "XBTUSD",
        "XBT/USD": "XBTUSD",
        "ETH/USD": "ETHUSD",
        "SOL/USD": "SOLUSD",
        "XRP/USD": "XRPUSD",
    }
    if normalized in aliases:
        return aliases[normalized]
    if "/" in normalized:
        return normalized.replace("/", "")
    return normalized


def _build_private_order_payload(args: argparse.Namespace) -> dict[str, str]:
    payload = {
        "nonce": str(int(time.time() * 1000)),
        "pair": _normalize_private_api_pair(args.pair),
        "type": args.side,
        "ordertype": args.order_type,
        "volume": args.volume,
        "validate": "true" if args.validate else "false",
    }
    if args.price:
        payload["price"] = args.price
    return payload


def _sign_kraken_request(*, api_path: str, payload: Mapping[str, str], api_secret: str) -> str:
    nonce = payload["nonce"]
    post_data = urlencode(payload)
    encoded = (nonce + post_data).encode("utf-8")
    message = api_path.encode("utf-8") + hashlib.sha256(encoded).digest()
    secret = base64.b64decode(api_secret)
    signature = hmac.new(secret, message, hashlib.sha512)
    return base64.b64encode(signature.digest()).decode("utf-8")


def _submit_private_add_order(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> tuple[int, dict[str, Any] | None, str]:
    api_key = env.get("KRAKEN_API_KEY", "").strip()
    api_secret = env.get("KRAKEN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return (
            2,
            None,
            "KRAKEN_API_KEY and KRAKEN_API_SECRET are required for Kraken CLI live access.",
        )
    if not args.validate and not _live_submit_enabled(env):
        return (
            2,
            None,
            "Live submission is blocked. Set "
            "KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true to send a real order.",
        )

    api_path = "/0/private/AddOrder"
    payload = _build_private_order_payload(args)
    headers = {
        "API-Key": api_key,
        "API-Sign": _sign_kraken_request(
            api_path=api_path,
            payload=payload,
            api_secret=api_secret,
        ),
    }
    api_url = env.get("KRAKEN_API_URL", _DEFAULT_KRAKEN_API_URL).rstrip("/")

    try:
        response = requests.post(
            f"{api_url}{api_path}",
            data=payload,
            headers=headers,
            timeout=float(env.get("KRAKEN_CLI_HTTP_TIMEOUT_SECONDS", "15")),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        return 124, None, str(exc)
    except requests.RequestException as exc:
        message = str(exc)
        exit_code = 75 if _is_retryable_failure(75, message) else 1
        return exit_code, None, message

    try:
        parsed = response.json()
    except ValueError:
        return 1, None, response.text

    errors = parsed.get("error") or []
    if errors:
        message = "; ".join(str(item) for item in errors)
        exit_code = 75 if _is_retryable_failure(75, message) else 1
        return exit_code, parsed, message
    return 0, parsed, ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        _validate_cli_inputs(
            pair=args.pair,
            side=args.side,
            order_type=args.order_type,
            volume=args.volume,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    env = os.environ
    exit_code, payload, error_message = _submit_private_add_order(args=args, env=env)
    if exit_code != 0:
        print(error_message, file=sys.stderr)
        return exit_code

    result = payload or {}
    print(
        json.dumps(
            {
                "status": "validated" if args.validate else "submitted",
                "validated": bool(args.validate),
                "live_submission": not args.validate,
                "pair": args.pair,
                "side": args.side,
                "type": args.order_type,
                "volume": args.volume,
                "client_order_id": args.clordid,
                "response": result.get("result", result),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
