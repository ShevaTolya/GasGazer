#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GasGazer — EIP-1559 помощник по газу для EVM-сетей.

Функции:
  - now:      Рекомендация maxFeePerGas и maxPriorityFeePerGas из последних блоков
  - when:     Исторические дешёвые часы (по последним N блокам)
  - threshold Порог "разумной" цены газа для economy/normal/fast
  - watch:    Онлайновое наблюдение за baseFee новых блоков

Требуется: Python 3.9+, requests, rich, pytz, python-dateutil
"""

import argparse
import time
from typing import List, Dict, Any, Tuple
import requests
from statistics import median
from datetime import datetime, timezone
from dateutil import tz
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def call_rpc(rpc: str, method: str, params: List[Any]) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        r = requests.post(rpc, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        return data["result"]
    except Exception as e:
        raise RuntimeError(f"RPC error for {method}: {e}")

def wei_to_gwei(wei: int) -> float:
    return wei / 1_000_000_000

def hex_to_int(x: str) -> int:
    return int(x, 16)

def get_fee_history(rpc: str, block_count: int, newest_block: str = "latest", reward_percentiles: List[float] = [10.0, 50.0, 90.0]) -> Dict[str, Any]:
    # eth_feeHistory(count, newestBlock, rewardPercentiles)
    return call_rpc(rpc, "eth_feeHistory", [hex(block_count), newest_block, reward_percentiles])

def get_block_number(rpc: str) -> int:
    return hex_to_int(call_rpc(rpc, "eth_blockNumber", []))

def format_gwei(x: float) -> str:
    # без лишних нулей, но стабильно читабельно
    if x >= 100:
        return f"{x:.0f} gwei"
    elif x >= 10:
        return f"{x:.1f} gwei"
    else:
        return f"{x:.2f} gwei"

def recommend_fees(rpc: str, lookback_blocks: int = 120, urgency: str = "normal", safety: float = 1.2) -> Tuple[float, float, float]:
    """
    Возвращает кортеж (base_fee_gwei, max_priority_gwei, max_fee_gwei)
    - baseFee: медиана последних базовых комиссий
    - priority: по p50 (с зажимами)
    - maxFee: base * safety + priority
    """
    if lookback_blocks < 10:
        lookback_blocks = 10

    fh = get_fee_history(rpc, lookback_blocks, "latest", [10.0, 50.0, 90.0])
    base_fees = [hex_to_int(bf) for bf in fh["baseFeePerGas"]]
    # baseFeePerGas длиной count+1, берём все, медиану
    base_fee_med_gwei = wei_to_gwei(median(base_fees))

    # reward — список массивов по блокам; берём p50
    rewards = fh.get("reward", [])
    p50_rewards_gwei = []
    for arr in rewards:
        if len(arr) >= 2:
            p50_rewards_gwei.append(wei_to_gwei(hex_to_int(arr[1])))
    priority_med = median(p50_rewards_gwei) if p50_rewards_gwei else 1.0

    # Базовые зажимы под тип срочности
    if urgency == "economy":
        priority = max(0.5, min(priority_med, 1.5))
        safety_mult = max(1.05, min(safety, 1.15))
    elif urgency == "fast":
        priority = max(2.0, min(priority_med * 1.5, 5.0))
        safety_mult = max(1.3, safety, 1.3)
    else:  # normal
        priority = max(1.0, min(priority_med * 1.2, 2.5))
        safety_mult = max(1.15, min(safety, 1.25))

    max_fee = base_fee_med_gwei * safety_mult + priority
    return base_fee_med_gwei, priority, max_fee

def cmd_now(args):
    base_fee, prio, max_fee = recommend_fees(args.rpc, args.lookback, args.urgency, args.safety)

    table = Table(title="GasGazer — рекомендации по комиссии (EIP-1559)", box=box.SIMPLE_HEAVY)
    table.add_column("Параметр", style="bold")
    table.add_column("Значение")
    table.add_row("Сетевой baseFee (медиана)", format_gwei(base_fee))
    table.add_row("Рекомендуемый maxPriorityFee", format_gwei(prio))
    table.add_row("Рекомендуемый maxFee", format_gwei(max_fee))
    table.add_row("Режим срочности", args.urgency)
    table.add_row("Окно анализа", f"{args.lookback} блоков")
    console.print(table)

    console.print(
        "\nПодсказка: установите в кошельке [bold]maxFeePerGas[/bold] и [bold]maxPriorityFeePerGas[/bold] как выше. "
        "Если транзакция не включается быстро — слегка повысить priority.\n"
    )

def cmd_threshold(args):
    # Рассчитаем типовые пороги на базе короткого окна + консервативные множители
    base_fee, prio, _ = recommend_fees(args.rpc, max(60, args.lookback), "normal", 1.2)

    # Экономичный/Нормальный/Быстрый пороги
    economy = base_fee * 1.05 + max(0.5, prio * 0.6)
    normal = base_fee * 1.15 + max(1.0, prio * 1.0)
    fast = base_fee * 1.35 + max(2.0, prio * 1.5)

    table = Table(title="GasGazer — пороговые значения газа", box=box.SIMPLE_HEAVY)
    table.add_column("Режим", style="bold")
    table.add_column("maxPriorityFee")
    table.add_column("maxFee (целевой порог)")
    table.add_row("economy", format_gwei(max(0.5, prio * 0.6)), format_gwei(economy))
    table.add_row("normal", format_gwei(max(1.0, prio * 1.0)), format_gwei(normal))
    table.add_row("fast", format_gwei(max(2.0, prio * 1.5)), format_gwei(fast))
    console.print(table)
    console.print("\nСмысл: отправляйте, когда текущая оценка ниже порога выбранного режима.\n")

def cmd_when(args):
    """
    Ищем дешёвые часы суток по истории baseFee.
    Идея: забираем M блоков, группируем по часу (UTC или локальному), берём медианы и показываем топ-слоты.
    """
    tz_out = tz.gettz(args.tz) if args.tz else timezone.utc
    newest = "latest"
    count = args.blocks

    fh = get_fee_history(args.rpc, count, newest, [50.0])
    base_fees = [hex_to_int(bf) for bf in fh["baseFeePerGas"]]
    # базовые отметки времени: oldestBlock известен, у feeHistory нет явных timestamps — запросим якорный и шаг
    # Сначала узнаем номер последнего блока и затем достанем пачкой timestamps для грубого выравнивания

    latest_block = get_block_number(args.rpc)
    # приблизим: позиции соответствуют интервалу [latest-count+1, latest]
    start_block = max(0, latest_block - len(base_fees) + 1)
    # Возьмём timestamps точечно (не для каждого), чтобы не долбить RPC: 32 равномерные выборки
    samples = 32
    step = max(1, len(base_fees) // samples)
    stamp_map: Dict[int, int] = {}
    for idx in range(0, len(base_fees), step):
        block_num = start_block + idx
        header = call_rpc(args.rpc, "eth_getBlockByNumber", [hex(block_num), False])
        if header and header.get("timestamp"):
            stamp_map[idx] = hex_to_int(header["timestamp"])

    # линейная интерполяция timestamp между сэмплами
    def ts_for_index(i: int) -> int:
        if i in stamp_map:
            return stamp_map[i]
        # найти левый и правый якорь
        left = max([k for k in stamp_map.keys() if k <= i] or [min(stamp_map.keys())])
        right = min([k for k in stamp_map.keys() if k >= i] or [max(stamp_map.keys())])
        if left == right:
            return stamp_map[left]
        # линейно
        t_left, t_right = stamp_map[left], stamp_map[right]
        ratio = (i - left) / (right - left)
        return int(t_left + (t_right - t_left) * ratio)

    # агрегируем по часу (локаль из args.tz)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, bf in enumerate(base_fees):
        ts = ts_for_index(i)
        dt = datetime.fromtimestamp(ts, tz=tz_out)
        hour_label = dt.strftime("%H:00")
        buckets[hour_label].append(wei_to_gwei(bf))

    hour_stats = []
    for hour, vals in buckets.items():
        if not vals:
            continue
        hour_stats.append((hour, median(vals), len(vals)))

    hour_stats.sort(key=lambda x: x[1])  # по возрастанию медианы baseFee

    table = Table(title=f"GasGazer — дешёвые часы за последние {count} блоков", box=box.SIMPLE_HEAVY)
    table.add_column("Час", style="bold")
    table.add_column("Медиана baseFee")
    table.add_column("Кол-во блоков")
    for hour, med, n in hour_stats[: min(6, len(hour_stats))]:
        table.add_row(hour, format_gwei(med), str(n))
    console.print(table)
    console.print(
        f"\nПодсказка: указанный час относится к зоне: [bold]{args.tz or 'UTC'}[/bold]. "
        "Планируйте неторопливые транзакции в эти окна.\n"
    )

def cmd_watch(args):
    console.print("[bold]Режим наблюдения за газом[/bold] — жмите Ctrl+C для выхода.")
    last_block = None
    try:
        while True:
            bn = get_block_number(args.rpc)
            if last_block is None or bn > last_block:
                fh = get_fee_history(args.rpc, 1, "latest", [50.0])
                base_fee = wei_to_gwei(hex_to_int(fh["baseFeePerGas"][-1]))
                console.print(f"Блок #{bn}  baseFee≈ {format_gwei(base_fee)}")
                last_block = bn
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        console.print("\nЗавершено.")

def build_parser():
    p = argparse.ArgumentParser(prog="gasgazer", description="GasGazer — помощник по газу (EIP-1559) для EVM.")
    p.add_argument("--rpc", required=True, help="JSON-RPC URL (например, https://mainnet.infura.io/v3/…)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_now = sub.add_parser("now", help="Рекомендовать maxFee/maxPriority прямо сейчас")
    p_now.add_argument("--lookback", type=int, default=120, help="Сколько последних блоков анализировать (по умолчанию 120)")
    p_now.add_argument("--urgency", choices=["economy", "normal", "fast"], default="normal", help="Срочность транзакции")
    p_now.add_argument("--safety", type=float, default=1.2, help="Множитель запаса для baseFee")
    p_now.set_defaults(func=cmd_now)

    p_thr = sub.add_parser("threshold", help="Пороговые значения газа для отправки")
    p_thr.add_argument("--lookback", type=int, default=90, help="Окно анализа блоков (по умолчанию 90)")
    p_thr.set_defaults(func=cmd_threshold)

    p_when = sub.add_parser("when", help="Исторически дешёвые часы отправки")
    p_when.add_argument("--blocks", type=int, default=4096, help="Сколько блоков анализировать (по умолчанию 4096)")
    p_when.add_argument("--tz", type=str, default=None, help="Временная зона (например, Europe/Madrid). По умолчанию UTC")
    p_when.set_defaults(func=cmd_when)

    p_watch = sub.add_parser("watch", help="Онлайновое наблюдение baseFee новых блоков")
    p_watch.add_argument("--interval", type=float, default=3.0, help="Интервал опроса, сек (по умолчанию 3)")
    p_watch.set_defaults(func=cmd_watch)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
