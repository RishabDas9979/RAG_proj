"""Data models, a synthetic dataset generator, and a loader for Kaggle's PaySim schema.

The synthetic generator is the only data this project has been exercised against end to end.
``load_paysim_csv`` follows the public PaySim column layout but was tested only on a small
sample written in that layout, because Kaggle was not reachable when this was built.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    country: str
    occupation: str
    pep: bool
    doc_expiry: date
    last_review: date
    expected_monthly_volume: float


@dataclass(frozen=True)
class Transaction:
    id: str
    customer_id: str
    ts: datetime
    channel: str        # cash | wire | transfer | card
    direction: str      # in | out
    amount: float
    counterparty_country: str
    counterparty_name: str


@dataclass(frozen=True)
class WatchlistEntry:
    id: str
    name: str
    list_name: str
    aliases: tuple[str, ...] = ()


@dataclass
class Dataset:
    customers: dict[str, Customer]
    transactions: list[Transaction]
    watchlist: list[WatchlistEntry]
    truth: dict[str, dict] = field(default_factory=dict)   # customer_id -> {"typology", "expected"}
    end: date = date(2026, 9, 30)
    _by_customer: dict[str, list[Transaction]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for t in sorted(self.transactions, key=lambda t: t.ts):
            self._by_customer.setdefault(t.customer_id, []).append(t)

    def transactions_for(self, customer_id: str) -> list[Transaction]:
        return self._by_customer.get(customer_id, [])


WATCHLIST = [
    WatchlistEntry("W001", "Zelmar Dravenko", "EXAMPLE-SANCTIONS", ("Z. Dravenko",)),
    WatchlistEntry("W002", "Orsolya Vantrek Mbeki", "EXAMPLE-SANCTIONS"),
    WatchlistEntry("W003", "Kasimir Oduya Brandt", "EXAMPLE-SANCTIONS"),
    WatchlistEntry("W004", "Talwyn Rhosyn Ashgrove", "EXAMPLE-SANCTIONS"),
]

_FIRST = ["Aarav", "Meera", "Kiran", "Lena", "Omar", "Sofia", "Daniel", "Priya", "Tomas", "Noor", "Hana",
          "Ivan", "Rhea", "Mateo", "Zara", "Arjun", "Elena", "Yusuf", "Mina", "Lucas", "Anika", "Felix",
          "Isla", "Rohan", "Nadia", "Jonas", "Leila", "Samir", "Clara", "Vikram"]
_LAST = ["Sharma", "Okafor", "Lindqvist", "Haddad", "Moreau", "Tanaka", "Petrov", "Alvarez", "Nair", "Kowalski",
         "Fischer", "Rahman", "Costa", "Iyer", "Bianchi", "Mehta", "Novak", "Silva", "Ibrahim", "Larsen",
         "Reddy", "Dubois", "Khan", "Santos", "Weber"]
_HOME = ["IN", "US", "GB", "DE", "SG", "AE", "CA", "AU"]
_OCCUPATIONS = ["teacher", "engineer", "shop owner", "nurse", "consultant", "driver", "designer", "accountant"]
_FOREIGN_LOW_RISK = ["FR", "NL", "JP", "CH", "SE"]

# Placeholder codes for "high-risk" jurisdictions. Replace with your institution's current list.
HIGH_RISK_COUNTRIES = frozenset({"XA", "XB", "XC"})

# (typology, expected action, how many customers)
_SCENARIOS = [
    ("structuring", "SAR", 3), ("rapid_movement", "SAR", 3), ("high_risk_geo", "SAR", 3),
    ("watchlist_match", "SAR", 3), ("velocity_spike", "EDD", 3), ("pep_large_wire", "EDD", 2),
    ("benign_pep", "EDD", 2), ("benign_large_wire", "EDD", 3), ("benign_large_cash", "NONE", 3),
    ("benign_expired_doc", "NONE", 3), ("watchlist_near_miss", "NONE", 2),
]


class _Builder:
    def __init__(self, rng: random.Random, start: date, days: int):
        self.rng, self.start, self.days = rng, start, days
        self.txns: list[Transaction] = []
        self._n = 0

    def add(self, cid: str, day: float, channel: str, direction: str, amount: float, country: str, cp: str):
        self._n += 1
        ts = datetime.combine(self.start, datetime.min.time()) + timedelta(days=day)
        self.txns.append(Transaction(f"T{self._n:06d}", cid, ts, channel, direction, round(amount, 2), country, cp))

    def normal(self, c: Customer):
        r = self.rng
        for _ in range(r.randint(8, 18)):
            day = r.uniform(0, self.days - 0.01)
            kind = r.choices(["card", "transfer", "wire", "cash"], [50, 20, 10, 20])[0]
            base = max(20.0, min(3000.0, c.expected_monthly_volume / 10 * r.lognormvariate(0, 0.5)))
            if kind == "card":
                self.add(c.id, day, "card", "out", base, c.country, f"Merchant {r.randint(100, 999)}")
            elif kind == "cash":
                self.add(c.id, day, "cash", r.choice(["in", "out"]), min(base, 2500), c.country, "Branch ATM")
            else:
                self.add(c.id, day, kind, r.choice(["in", "out"]), base, c.country, f"Party {r.randint(100, 999)}")


def generate_dataset(seed: int = 7, n_customers: int = 200, end: date = date(2026, 9, 30), days: int = 60,
                     evasive: bool = False) -> Dataset:
    """Create customers, transactions and ground-truth labels with injected AML typologies.

    With ``evasive=True`` the suspicious customers behave to avoid the rules' exact thresholds
    (looser structuring amounts spread over more days, outflows split past the 48-hour window,
    smaller high-risk wires). The labels stay the same, so recall drops where the rules are weak.
    """
    if n_customers < 40:
        raise ValueError("n_customers must be at least 40 to fit every scenario")
    rng = random.Random(seed)
    start = end - timedelta(days=days - 1)
    b = _Builder(rng, start, days)

    names: set[str] = set()
    customers: dict[str, Customer] = {}
    for i in range(1, n_customers + 1):
        while True:
            name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            if name not in names:
                names.add(name)
                break
        cid = f"C{i:04d}"
        customers[cid] = Customer(
            cid, name, rng.choice(_HOME), rng.choice(_OCCUPATIONS), False,
            end + timedelta(days=rng.randint(60, 1500)), end - timedelta(days=rng.randint(30, 300)),
            float(round(rng.uniform(2000, 15000), -2)),
        )

    ids = list(customers)
    rng.shuffle(ids)
    truth: dict[str, dict] = {}
    cursor = 0
    variants = iter(["Zelmar Dravenko", "Dravenko Zelmar", "Zelmar Dravenco", "Orsolya Vantrek-Mbeki"])
    near_misses = iter(["Kasim Oduya", "Zelmar Dravis"])

    def replace(cid: str, **changes):
        c = customers[cid]
        customers[cid] = Customer(**{**c.__dict__, **changes})

    for typology, expected, count in _SCENARIOS:
        for _ in range(count):
            cid = ids[cursor]
            cursor += 1
            truth[cid] = {"typology": typology, "expected": expected}
            c = customers[cid]
            b.normal(c)
            d0 = rng.uniform(5, days - 12)
            if typology == "structuring":
                n, lo, step = (6, 6500, 1.6) if evasive else (5, 8500, 1.0)
                for k in range(n):
                    b.add(cid, d0 + k * step + rng.random() * 0.5, "cash", "in", rng.uniform(lo, 9900), c.country, "Branch deposit")
            elif typology == "rapid_movement":
                amount = rng.uniform(55000, 70000)
                b.add(cid, d0, "wire", "in", amount, rng.choice(_FOREIGN_LOW_RISK), "Holdings A")
                if evasive:   # three smaller outflows over ~60 hours, 85% of the inflow in total
                    for k, when in enumerate((0.8, 1.6, 2.5)):
                        b.add(cid, d0 + when, "wire", "out", amount * 0.85 / 3, rng.choice(_FOREIGN_LOW_RISK), f"Holdings B{k}")
                else:
                    b.add(cid, d0 + 0.4, "wire", "out", amount * rng.uniform(0.94, 0.98), rng.choice(_FOREIGN_LOW_RISK), "Holdings B")
            elif typology == "high_risk_geo":
                lo, hi = (3500, 4500) if evasive else (7000, 8500)
                for k in range(4):
                    b.add(cid, d0 + k * 1.2, "wire", rng.choice(["in", "out"]), rng.uniform(lo, hi),
                          rng.choice(sorted(HIGH_RISK_COUNTRIES)), "Trading Co")
            elif typology == "velocity_spike":
                total = max(30000.0, 4 * c.expected_monthly_volume)
                for k in range(20):
                    b.add(cid, d0 + k * 0.25, "transfer", rng.choice(["in", "out"]), total / 20, c.country, f"Party {k}")
            elif typology == "pep_large_wire":
                replace(cid, pep=True)
                b.add(cid, d0, "wire", "in", rng.uniform(28000, 35000), c.country, "Consulting fee")
            elif typology == "watchlist_match":
                replace(cid, name=next(variants))
            elif typology == "watchlist_near_miss":
                replace(cid, name=next(near_misses))
            elif typology == "benign_pep":
                replace(cid, pep=True)
            elif typology == "benign_large_wire":
                b.add(cid, d0, "wire", "in", rng.uniform(38000, 45000), c.country, "Property sale")
            elif typology == "benign_large_cash":
                b.add(cid, d0, "cash", "in", rng.uniform(10500, 13000), c.country, "Vehicle sale")
            elif typology == "benign_expired_doc":
                replace(cid, doc_expiry=end - timedelta(days=rng.randint(5, 120)))

    for cid in ids[cursor:]:
        b.normal(customers[cid])
    return Dataset(customers, b.txns, list(WATCHLIST), truth, end)


_PAYSIM_MAP = {  # type -> (channel, direction)
    "CASH_IN": ("cash", "in"), "CASH_OUT": ("cash", "out"), "TRANSFER": ("transfer", "out"),
    "PAYMENT": ("card", "out"), "DEBIT": ("transfer", "out"),
}


def load_paysim_csv(path: str | Path, max_rows: int | None = None, base: datetime = datetime(2026, 1, 1),
                    end: date | None = None) -> Dataset:
    """Load Kaggle's PaySim layout (step,type,amount,nameOrig,...,nameDest,...,isFraud).

    ``step`` is one hour. Customers are the ``nameOrig`` accounts; ``isFraud`` becomes the label
    (expected action SAR). PaySim has no names, countries, PEP flags or ID documents, so those are
    filled with neutral defaults, and expected monthly volume is estimated from observed activity.
    """
    txns: list[Transaction] = []
    truth: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh), 1):
            if max_rows and n > max_rows:
                break
            channel, direction = _PAYSIM_MAP.get(row["type"], ("transfer", "out"))
            cid = row["nameOrig"]
            txns.append(Transaction(
                f"P{n:08d}", cid, base + timedelta(hours=int(row["step"])), channel, direction,
                float(row["amount"]), "XX", row["nameDest"],
            ))
            if row.get("isFraud") == "1":
                truth[cid] = {"typology": "paysim_fraud", "expected": "SAR"}
    last = max((t.ts for t in txns), default=base)
    end = end or last.date()
    totals: dict[str, float] = {}
    for t in txns:
        totals[t.customer_id] = totals.get(t.customer_id, 0.0) + t.amount
    span_months = max(1.0, (last - base).days / 30)
    customers = {
        cid: Customer(cid, cid, "XX", "unknown", False, end + timedelta(days=365), end - timedelta(days=90),
                      max(1.0, total / span_months))
        for cid, total in totals.items()
    }
    for cid in customers:
        truth.setdefault(cid, {"typology": "none", "expected": "NONE"})
    return Dataset(customers, txns, list(WATCHLIST), truth, end)
